import torch
import torch.fx as fx
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import copy
from torch.fx.node import Node
from ..utils.logger import get_logger
from ..utils.graph_utils import get_output_node

logger = get_logger(__name__)

class ActivationRecomputation:
    """
    A PyTorch FX Pass for performing Activation Recomputation optimization.
    This optimization aims to reduce memory consumption by recomputing certain intermediate activations
    during the backward pass, instead of storing them throughout the forward pass.

    This Pass strictly follows a three-step logic:
    1. Identify the nodes to be recomputed and their upstream dependencies in the forward graph.
    2. Modify the backward graph to insert the recomputation path.
    3. Modify the forward graph to adjust its outputs to support the recomputation flow and eliminate dead code.
    """
    def __init__(self, fw_gm: fx.GraphModule = None, bw_gm: fx.GraphModule = None):
        """
        Initializes the ActivationRecomputationPass.

        Args:
            fw_gm (fx.GraphModule, optional): The forward computation graph module. Defaults to None.
            bw_gm (fx.GraphModule, optional): The backward computation graph module. Defaults to None.
        """
        self.fw_gm = fw_gm
        self.bw_gm = bw_gm
        if fw_gm and bw_gm:
            self.re_init_maps()
        else:
            self.fw_name_to_node: Dict[str, fx.Node] = {}
            self.bw_name_to_node: Dict[str, fx.Node] = {}

    def re_init_maps(self):
        """
        Re-initializes the mapping from node names to node objects.

        Call this method after the graph structure has been modified (e.g., node names changed,
        nodes added/deleted) to ensure that the internal `fw_name_to_node` and `bw_name_to_node`
        maps are up-to-date and accurate. This is crucial for subsequent operations that
        look up nodes by name.

        Raises:
            ValueError: If `fw_gm` or `bw_gm` have not been set before calling.
        """
        if not self.fw_gm or not self.bw_gm:
            raise ValueError("fw_gm and bw_gm must be set before re-initializing the maps.")
        self.fw_name_to_node = {n.name: n for n in self.fw_gm.graph.nodes}
        self.bw_name_to_node = {n.name: n for n in self.bw_gm.graph.nodes}
        logger.debug("Maps have been re-initialized.")

    def run(self, activations_to_recompute_name: List[str]) -> Tuple[fx.GraphModule, fx.GraphModule]:
        """
        The main entry point for executing the activation recomputation optimization.

        This method coordinates the entire optimization process, including identifying the
        subgraph to be recomputed, modifying the backward graph to insert recomputation logic,
        adjusting the forward graph's output, and performing final graph cleanup and compilation.

        Args:
            activations_to_recompute_name (List[str]): A list of strings containing the names of
                                                         activation nodes in the forward graph that
                                                         need to be recomputed to save memory.

        Returns:
            Tuple[fx.GraphModule, fx.GraphModule]: The optimized (forward graph module, backward graph module).
        """
        # 0. 筛选出真正可重计算的激活（FW output ∩ BW placeholder ∩ 用户请求）
        true_activations_to_recompute_name = self._get_true_recomputation_candidates(activations_to_recompute_name)
        # 1. 确定重计算子图（BFS 向上追溯依赖链）及更新后的 FW 输出节点名集合
        sorted_nodes_to_insert, activation_node_names = self._identify_nodes_to_change(true_activations_to_recompute_name)

        # 1.5 补全子图中需要、但 BW 图中尚不存在的 FW primal placeholder
        self._ensure_primals_in_bw(sorted_nodes_to_insert, activation_node_names)

        # 2. 修改反向图：将重计算子图插入 BW，替换原有 saved placeholder
        logger.debug("True activations to recompute: %s", true_activations_to_recompute_name)
        logger.debug("Sorted nodes to insert: %s", sorted_nodes_to_insert)
        self._modify_backward_graph(true_activations_to_recompute_name, sorted_nodes_to_insert)

        # 3. 修改前向图：更新输出节点，执行死代码消除
        self._modify_forward_graph(activation_node_names, true_activations_to_recompute_name)
        self.fw_gm = self.rebuild_graph(self.fw_gm)
        self.bw_gm = self.rebuild_graph(self.bw_gm)

        # 4. 重新编译两张图
        for gm in [self.fw_gm, self.bw_gm]:
            gm.recompile()
            gm.graph.lint()

        logger.info("Activation Recomputation Pass executed successfully.")
        return self.fw_gm, self.bw_gm

    def _ensure_primals_in_bw(self, sorted_nodes_to_insert: List[Node], activation_node_names: set) -> None:
        """
        AOT Autograd may not save every FW primal (placeholder) to the BW graph
        if it was not needed for gradient computation in the original graph.
        When we recompute a subgraph in BW, that subgraph may reference such
        primals directly.  If lookup_fn_permissive cannot resolve them, it returns
        None, causing "got None for argument #0" at runtime.

        This method pre-processes sorted_nodes_to_insert:
          - Finds every FW placeholder input that is NOT yet a BW placeholder.
          - Adds it as a new BW placeholder (inserted before the first tangent
            placeholder so the calling-convention ordering stays consistent).
          - Adds its name to activation_node_names so _modify_forward_graph
            includes it in the FW output (making it available at runtime).
        """
        # Locate insertion point: just before the first tangent placeholder.
        first_tangent = None
        last_saved_ph = None
        for n in self.bw_gm.graph.nodes:
            if n.op == 'placeholder':
                if n.name.startswith('tangents_'):
                    if first_tangent is None:
                        first_tangent = n
                else:
                    last_saved_ph = n

        added: set = set()
        for fw_node in sorted_nodes_to_insert:
            for input_node in fw_node.all_input_nodes:
                if (input_node.op == 'placeholder'
                        and input_node.name not in self.bw_name_to_node
                        and input_node.name not in added):
                    added.add(input_node.name)
                    # Keep this primal in the FW output.
                    activation_node_names.add(input_node.name)
                    logger.debug(
                        f"[_ensure_primals_in_bw] FW primal '{input_node.name}' "
                        f"is needed by recomputed nodes but absent from BW graph; "
                        f"adding as new BW placeholder."
                    )
                    # Create the new BW placeholder.
                    if first_tangent is not None:
                        with self.bw_gm.graph.inserting_before(first_tangent):
                            new_ph = self.bw_gm.graph.placeholder(input_node.name)
                    elif last_saved_ph is not None:
                        with self.bw_gm.graph.inserting_after(last_saved_ph):
                            new_ph = self.bw_gm.graph.placeholder(input_node.name)
                    else:
                        raise RuntimeError(
                            f"Cannot find a valid insertion point for new BW "
                            f"placeholder '{input_node.name}'"
                        )
                    new_ph.meta = copy.copy(input_node.meta)
                    self.bw_name_to_node[new_ph.name] = new_ph
                    # Move the "last_saved_ph" pointer so subsequent insertions
                    # remain in order.
                    if first_tangent is None:
                        last_saved_ph = new_ph

        if added:
            logger.info(
                f"[_ensure_primals_in_bw] Added {len(added)} missing FW primals "
                f"as BW placeholders: {sorted(added)}"
            )

    def _get_true_recomputation_candidates(self, activations_to_recompute_name) -> Set[str]:
        """
        Implements the logic to find the precise set of activation names that 
        are candidates for recomputation in a distributed setting.
        """
        # Step 1: In the forward graph, find all "candidate" activations (all outputs).
        forward_outputs_names = set()
        for node in self.fw_gm.graph.nodes:
            if node.op == 'output':
                for input_node in node.all_input_nodes:
                    if input_node.op != 'placeholder':
                        forward_outputs_names.add(input_node.name)
                break  # Assuming a single output node

        # Step 2: In the backward graph, get all placeholder names.
        backward_local_placeholders_names = set()
        for node in self.bw_gm.graph.nodes:
            if node.op == 'placeholder':
                backward_local_placeholders_names.add(node.name)

        # Step 3: Take the intersection to get the final result.
        true_candidates = forward_outputs_names.intersection(backward_local_placeholders_names)
        logger.debug(f"Candidates after intersection with BW placeholders: {true_candidates}")
        
        true_candidates = true_candidates.intersection(activations_to_recompute_name)
        logger.debug(f"Candidates after intersection with user request: {true_candidates}")
        
        logger.debug(f"FW Outputs ({len(forward_outputs_names)}): {forward_outputs_names}")
        logger.debug(f"BW Local Placeholders ({len(backward_local_placeholders_names)}): {backward_local_placeholders_names}")
        logger.debug(f"True Recomputation Candidates ({len(true_candidates)}): {true_candidates}")

        return true_candidates

    def _identify_nodes_to_change(self, activations_to_recompute: Set[str]) -> Tuple[List[fx.Node], Set[str]]:
        """
        Identifies all nodes in the forward graph that need to be recomputed.
        """
        logger.debug("--- 1. Start identifying nodes to change ---")

        nodes_to_change: Set[fx.Node] = set()
        activation_node_names = set()

        output_node = get_output_node(self.fw_gm)
        if output_node:
            for input_n in output_node.all_input_nodes:
                activation_node_names.add(input_n.name)
                
        logger.debug(f"Defined {len(activation_node_names)} activation boundaries: {list(activation_node_names)}")
        
        # Remove the activations to be recomputed from the boundaries
        activation_node_names.difference_update(activations_to_recompute)
        logger.debug(f"Updated {len(activation_node_names)} activation boundaries after removing candidates: {list(activation_node_names)}")
            
        # 2. Use a queue for a breadth-first upstream traversal.
        queue = deque()
        visited = set()

        for act_name in activations_to_recompute:
            if act_name in self.fw_name_to_node:
                start_node = self.fw_name_to_node[act_name]
                if start_node not in visited:
                    queue.append(start_node)
                    visited.add(start_node)
            else:
                logger.warning(f"Activation to be recomputed '{act_name}' does not exist in the forward graph.")

        while queue:
            current_node = queue.popleft()

            if current_node.name in activation_node_names:
                continue

            if current_node.op != 'placeholder':
                nodes_to_change.add(current_node)
                logger.debug(f"Processing node: {current_node}, inputs: {current_node.all_input_nodes}")
                for input_node in current_node.all_input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.append(input_node)
                        
        logger.debug(f"Identification complete, found {len(nodes_to_change)} nodes to be changed: {[n.name for n in sorted(list(nodes_to_change), key=lambda x: x.name)]}")

        # 子图规模检查：若重计算节点超过 FW 图计算节点总数的 50% 则发出警告
        total_fw_compute = sum(
            1 for n in self.fw_gm.graph.nodes if n.op not in ('placeholder', 'output')
        )
        if total_fw_compute > 0:
            recompute_ratio = len(nodes_to_change) / total_fw_compute
            logger.info(
                f"[Recompute] 子图规模: {len(nodes_to_change)}/{total_fw_compute} "
                f"({recompute_ratio:.0%}) FW 计算节点将被重计算。"
            )
            if recompute_ratio > 0.5:
                logger.warning(
                    f"[Recompute] 重计算子图覆盖了 {recompute_ratio:.0%} 的 FW 计算节点，"
                    f"可能导致重计算开销过高。建议检查是否存在未保存的中间激活导致 BFS 越界向上追溯。"
                )
        
        activation_node_names.update(
            node.name for node in nodes_to_change if node.op == 'placeholder'
        )
        logger.debug(f"Identification complete, found {len(activation_node_names)} activation nodes: {list(activation_node_names)}")
        
        try:
            logger.debug(f"nodes_to_change: {nodes_to_change}")
            sorted_nodes_to_insert = self._topological_sort(nodes_to_change)
            logger.debug(f"Global topological sort complete, {len(sorted_nodes_to_insert)} nodes in total.")
        except RuntimeError as e:
            logger.error("Topological sort error: %s", e)
            raise RuntimeError(
                f"重计算子图拓扑排序失败，可能存在环路。请检查 activations_to_recompute 的合法性。"
            ) from e
            
        return sorted_nodes_to_insert, activation_node_names

    def _modify_forward_graph(self, target_output_names: List[str], activations_to_recompute_name):
        """
        Completely resets the forward graph's output based on a given list of names.
        """
        logger.debug("--- 2. Start resetting the forward graph's output based on the given list ---")
            
        fw_output_node = get_output_node(self.fw_gm)
        if not fw_output_node:
            logger.warning("Forward graph has no output node, skipping modification.")
            return

        target_nodes = []
        for name in target_output_names:
            if name in self.fw_name_to_node:
                target_nodes.append(self.fw_name_to_node[name])
            else:
                logger.warning(f"Target output '{name}' does not exist in the forward graph and will be ignored.")

        final_outputs = list(dict.fromkeys(target_nodes))
        fw_output_node.args = (tuple(final_outputs),)

        logger.debug(f"Forward graph output has been reset. The new output contains {len(final_outputs)} nodes: {[n.name for n in final_outputs]}")
        
        logger.debug("--- 3. Start cleaning up unused nodes in the forward graph ---")

        worklist = deque()
        for node_name in activations_to_recompute_name:
            if node_name in self.fw_name_to_node:
                worklist.append(self.fw_name_to_node[node_name])

        deleted_node_count = 0
        visited = set()

        while worklist:
            node = worklist.popleft()
            if node in visited:
                continue
            visited.add(node)

            if len(node.users) == 0 and node.op not in {'output', 'placeholder'}:
                logger.debug(f"Deleting node '{node.name}' (op: {node.op}) because it has no users.")

                for input_node in node.all_input_nodes:
                    if input_node not in visited:
                        worklist.append(input_node)

                self.fw_gm.graph.erase_node(node)
                deleted_node_count += 1

        logger.debug(f"Manual cleanup complete. Removed {deleted_node_count} unused nodes.")

    def _modify_backward_graph(self, activations_to_replace: List[str], sorted_nodes_to_insert: List[Node]):
        """
        Inserts recomputed nodes into the backward graph.
        """
        placeholders_to_replace = {}
        for act_name in activations_to_replace:
            placeholder = self.bw_name_to_node.get(act_name)
            if placeholder and placeholder.users:
                placeholders_to_replace[act_name] = placeholder
                placeholder.name = f"_{act_name}_old_placeholder"
                
        self.re_init_maps()
        logger.debug(f"Batch renaming of {len(placeholders_to_replace)} old placeholders complete.")
        
        if not placeholders_to_replace:
            return

        global_node_map: Dict[fx.Node, fx.Node] = {}

        # lookup_fn_permissive 引用 global_node_map（循环中会被填充），
        # 定义在循环外以避免每次迭代都创建新函数对象。
        def lookup_fn_permissive(n: fx.Node) -> Optional[fx.Node]:
            if n in global_node_map:
                return global_node_map[n]
            return self.bw_name_to_node.get(n.name)

        # 预构建 BW 节点位置索引，避免 O(N²) 线性扫描
        bw_node_position: Dict[fx.Node, int] = {
            node: i for i, node in enumerate(self.bw_gm.graph.nodes)
        }

        for node_to_create in sorted_nodes_to_insert:
            logger.debug(f"[Loop] Processing original forward node: {node_to_create.op} '{node_to_create.name}'")

            new_name = node_to_create.name + '_fw'

            if new_name in self.bw_name_to_node:
                if not any(new_name == p.name for p in placeholders_to_replace.values()):
                    raise RuntimeError(
                        f"Naming conflict! Node '{new_name}' already exists in the backward graph and is not a placeholder to be replaced."
                    )

            new_args = fx.node.map_arg(node_to_create.args, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)
            new_kwargs = fx.node.map_arg(node_to_create.kwargs, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)

            # 用位置索引 O(1) 找到最晚的依赖节点，避免 O(N²) 全图扫描
            all_input_nodes: List[fx.Node] = []
            def _collect(a):
                if isinstance(a, fx.Node):
                    all_input_nodes.append(a)
            fx.node.map_arg(new_args, _collect)
            fx.node.map_arg(new_kwargs, _collect)

            latest_dependency_node = max(
                (n for n in all_input_nodes if n in bw_node_position),
                key=lambda n: bw_node_position[n],
                default=None,
            )
            
            if latest_dependency_node:
                insertion_point = latest_dependency_node
            else:
                # 无输入依赖：插在最后一个 placeholder 之后
                insertion_point = max(
                    (n for n in bw_node_position if n.op == 'placeholder'),
                    key=lambda n: bw_node_position[n],
                    default=None,
                )

            if insertion_point is None:
                raise RuntimeError("Could not find any valid insertion point for the node!")

            with self.bw_gm.graph.inserting_after(insertion_point):
                meta_copy = copy.copy(node_to_create.meta)
                new_node = self.bw_gm.graph.create_node(
                    name=new_name, 
                    op=node_to_create.op, 
                    target=node_to_create.target,
                    args=new_args, 
                    kwargs=new_kwargs
                )
                new_node.meta = meta_copy
                logger.debug("> Successfully created and inserted new node: '%s'", new_node.name)

                global_node_map[node_to_create] = new_node
                self.bw_name_to_node[new_node.name] = new_node
                # 插入新节点后更新位置索引，保证后续迭代的 max() 仍然正确
                bw_node_position[new_node] = max(bw_node_position.values()) + 1

        replaced_count = 0
        for act_name, old_placeholder_node in placeholders_to_replace.items():
            original_fw_node = self.fw_name_to_node.get(act_name)
            final_recomputed_node = global_node_map.get(original_fw_node)
            
            if final_recomputed_node:
                old_placeholder_node.replace_all_uses_with(final_recomputed_node)
                self.bw_gm.graph.erase_node(old_placeholder_node)
                replaced_count += 1
            else:
                raise RuntimeError(
                    f"[_modify_backward_graph] 无法找到激活 '{act_name}' 对应的重计算节点。"
                    f"请检查 sorted_nodes_to_insert 是否包含该激活的完整依赖链。"
                )
        
        logger.debug(f"Successfully replaced {replaced_count} / {len(placeholders_to_replace)} old placeholders.")
        self.re_init_maps()

    def _topological_sort(self, nodes_to_sort: Set[fx.Node]) -> List[fx.Node]:
        """
        Performs a topological sort on a given set of `fx.Node`s.
        """
        logger.debug(f"Starting topological sort for {len(nodes_to_sort)} nodes.")

        sorted_nodes = []
        in_degree = {node: 0 for node in nodes_to_sort}
        users_map = {node: [] for node in nodes_to_sort}

        for u in nodes_to_sort:
            for v in u.users:
                if v in nodes_to_sort:
                    in_degree[v] += 1
                    users_map[u].append(v)

        queue = deque([node for node in nodes_to_sort if in_degree[node] == 0])
        
        while queue:
            u = queue.popleft()
            sorted_nodes.append(u)
            for v in users_map[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_nodes) != len(nodes_to_sort):
            logger.error(f"Topological sort failed, cycle detected. Nodes to sort: {len(nodes_to_sort)}, nodes sorted: {len(sorted_nodes)}")
            raise RuntimeError("A cycle exists in the subgraph of nodes to be inserted, topological sort is not possible!")
            
        logger.debug(f"Topological sort complete. Sorted result (partial): {[n.name for n in sorted_nodes[:min(5, len(sorted_nodes))]]}...")
        return sorted_nodes

    def rebuild_graph(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Rebuilds a GraphModule to ensure nodes are topologically sorted.
        """
        all_nodes = list(gm.graph.nodes)
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        adj = [[] for _ in all_nodes]
        in_degree = [0] * len(all_nodes)
        
        for i, node in enumerate(all_nodes):
            for dep_node in node.all_input_nodes:
                if dep_node in node_map:
                    dep_idx = node_map[dep_node]
                    adj[dep_idx].append(i)
                    in_degree[i] += 1

        queue = deque([i for i, deg in enumerate(in_degree) if deg == 0])
        sorted_nodes = []
        while queue:
            u_idx = queue.popleft()
            sorted_nodes.append(all_nodes[u_idx])
            for v_idx in adj[u_idx]:
                in_degree[v_idx] -= 1
                if in_degree[v_idx] == 0:
                    queue.append(v_idx)

        if len(sorted_nodes) != len(all_nodes):
            raise RuntimeError("Cycle detected in the graph, cannot topologically sort!")

        new_graph = fx.Graph()
        old_to_new_map: Dict[Node, Node] = {}

        def arg_transform(n: Node) -> Node:
            return old_to_new_map[n]

        for old_node in sorted_nodes:
            new_node = new_graph.node_copy(old_node, arg_transform)
            old_to_new_map[old_node] = new_node
            
        new_gm = fx.GraphModule(gm, new_graph)
        return new_gm