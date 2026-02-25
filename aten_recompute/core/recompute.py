import torch
import torch.fx as fx
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import copy
from ..Comm.hccl import *
from torch.fx.node import Node
# Global debug switch
DEBUG = False # Set DEBUG to True to enable all debug prints

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
        if DEBUG: # Add DEBUG check
            print("[DEBUG] Maps have been re-initialized.")

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
        #0
        true_activations_to_recompute_name=self._get_true_recomputation_candidates(activations_to_recompute_name)
        # 1. Identify the nodes to be changed (the subgraph leading to the recomputation targets); these should point to the output.
        sorted_nodes_to_insert ,activation_node_names= self._identify_nodes_to_change(true_activations_to_recompute_name)
        # sorted_nodes_to_insert=self.rename_and_relink_nodes(list(sorted_nodes_to_insert))
        # Pass the nodes to be recomputed and the nodes that need to be changed.
        # 3. Modify the backward graph.
        # print("---------------------------------------------------")
        # print(true_activations_to_recompute_name)
        # print(sorted_nodes_to_insert)
        # print("*******************************************************")

        self._modify_backward_graph(true_activations_to_recompute_name, sorted_nodes_to_insert)
        # 2. Modify the forward graph.
        # Pass the now-modified activation node names, use built-in dead code elimination.
        self._modify_forward_graph(activation_node_names, true_activations_to_recompute_name)
        self.fw_gm = self.rebuild_graph(self.fw_gm)
        self.bw_gm = self.rebuild_graph(self.bw_gm)
        # 4. Clean up the graphs and recompile.
        for gm in [self.fw_gm, self.bw_gm]:
            gm.recompile()
            gm.graph.lint()

        if DEBUG: # Add DEBUG check
            print("\n Pass executed successfully.")
        return self.fw_gm, self.bw_gm
    def _get_true_recomputation_candidates(self,activations_to_recompute_name) -> Set[str]:
        """
        Implements the logic to find the precise set of activation names that 
        are candidates for recomputation in a distributed setting.
        """
        # Step 1: In the forward graph, find all "candidate" activations (all outputs).
        forward_outputs_names = set()
        for node in self.fw_gm.graph.nodes:
            if node.op == 'output':
                for input_node in node.all_input_nodes :
                    if input_node.op!='placeholder':
                        forward_outputs_names.add(input_node.name)
                break  # Assuming a single output node

        # Step 2: In the backward graph, get all placeholder names.
        backward_local_placeholders_names = set()
        for node in self.bw_gm.graph.nodes:
            if node.op == 'placeholder':
                    backward_local_placeholders_names.add(node.name)

        # Step 3: Take the intersection to get the final result.
        true_candidates = forward_outputs_names.intersection(backward_local_placeholders_names)
        if DEBUG:
            print(true_candidates)
        true_candidates=true_candidates.intersection(activations_to_recompute_name)
        if DEBUG:
            print(true_candidates)
        
        if DEBUG:
            print(f"[DEBUG] FW Outputs ({len(forward_outputs_names)}): {forward_outputs_names}")
            print(f"[DEBUG] BW Local Placeholders ({len(backward_local_placeholders_names)}): {backward_local_placeholders_names}")
            print(f"[DEBUG] True Recomputation Candidates ({len(true_candidates)}): {true_candidates}")

        return true_candidates
    def _identify_nodes_to_change(self, activations_to_recompute: List[str]) -> Tuple[Set[fx.Node], Set[str]]: # Fix: Retain the original Tuple return type
        """
        Identifies all nodes in the forward graph that need to be recomputed (i.e., need to be
        "changed" and inserted into the backward graph).

        This method uses a Breadth-First Search (BFS) starting from the user-specified
        `activations_to_recompute` nodes, traversing upstream through the forward graph. It
        collects all intermediate nodes that form the path from the graph's original inputs
        to these activations. The traversal stops at the graph's original inputs (`placeholder`
        nodes) or at "activation boundary" nodes related to the original `output` node.

        Args:
            activations_to_recompute (List[str]): A list of activation node names specified by the
                                                   user for recomputation.

        Returns:
            Tuple[Set[fx.Node], Set[str]]:
                - nodes_to_change (Set[fx.Node]): The set of nodes to be copied from the forward
                                                  graph and inserted into the backward graph for
                                                  recomputation. This does not include original
                                                  input nodes or the recomputation endpoints themselves.
                - activation_node_names (Set[str]): Contains all original output dependencies
                                                   (after comparison with the recomputation list)
                                                   and the names of `placeholder` nodes preserved
                                                   for recomputation. This set will be used to adjust
                                                   the final output of the forward graph.
        """
        if DEBUG: # Add DEBUG check
            print("--- 1. Start identifying nodes to change ---")
            # ==========================================================
            # Add logic here to print the parameters of the Output node
            # ==========================================================
        output_node = None
        # Traverse the graph nodes from back to front to find the first 'output' node
        for node in reversed(self.fw_gm.graph.nodes):
            if node.op == 'output':
                output_node = node
                break
        # ==========================================================
        nodes_to_change: Set[fx.Node] = set()

        # 1. Find all user-defined "activation nodes" (i.e., nodes that are inputs to the output).
        # The `activation_node_names` here are used to define the stopping boundary for the BFS,
        # and will ultimately be the set of node names for modifying the forward graph's output.
        activation_node_names = set()
        
        output_node = None # Re-find the output_node to ensure correctness
        for node in reversed(self.fw_gm.graph.nodes):
            if node.op == 'output':
                output_node = node
                break
        if output_node:
            for input_n in output_node.all_input_nodes:
                # (Modification 1: Only add the node's name)
                activation_node_names.add(input_n.name)
        if DEBUG: # Add DEBUG check
            print(f"Defined {len(activation_node_names)} activation boundaries: {list(activation_node_names)}")
        # Remove the activations to be recomputed from the boundaries, as they are the
        # starting points of the BFS, not the stopping conditions.
        activation_node_names.difference_update(activations_to_recompute)
        if DEBUG: # Add DEBUG check
            print(f"Defined {len(activation_node_names)} activation boundaries: {list(activation_node_names)}")
            
        # 2. Use a queue for a breadth-first upstream traversal.
        queue = deque() # Queue for nodes to visit
        visited = set() # Set of visited nodes

        # Initialize the queue, adding all activation nodes to be recomputed as the starting points for BFS.
        for act_name in activations_to_recompute:
            if act_name in self.fw_name_to_node:
                start_node = self.fw_name_to_node[act_name]
                if start_node not in visited:
                    queue.append(start_node)
                    visited.add(start_node)
            else:
                if DEBUG: # Add DEBUG check
                    print(f"Warning: Activation to be recomputed '{act_name}' does not exist in the forward graph.")

        # --- Traversal logic correction ---
        while queue:
            current_node = queue.popleft()
            
            # Stopping condition: The node is one of the remaining activation values of the graph.
            if current_node.name in activation_node_names:
                continue
            
            # If the node passes all stopping condition checks, then it is a node that needs to be changed.
            # Key modification: Only add the node to the result set if it is not an initial node.
            # if current_node.name not in activations_to_recompute: # This line is commented out, preserving the original intent
            #     nodes_to_change.add(current_node)
            nodes_to_change.add(current_node) # The original code logic is to add it directly
            if DEBUG: # Add DEBUG check
                print(current_node, current_node.all_input_nodes)
            # Add the upstream neighbors of the current node to the queue to continue the traversal.
            if current_node.op != 'placeholder':
                for input_node in current_node.all_input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.append(input_node)
        if DEBUG : # Add DEBUG check
            print(f"Identification complete, found {len(nodes_to_change)} nodes to be changed: {[n.name for n in sorted(list(nodes_to_change), key=lambda x: x.name)]}")
        
        # Filter for placeholders in nodes_to_change and add their [names] directly to the activation_node_names set.
        # This is to ensure that the activation_node_names eventually passed to _modify_forward_graph contains all necessary outputs.
        activation_node_names.update(
            node.name for node in nodes_to_change if node.op == 'placeholder'
        )
        if DEBUG: # Add DEBUG check
            print(f"Identification complete, found {len(activation_node_names)} activation nodes: {list(activation_node_names)}")
        # Return the affected nodes and the current activation values.
        try:
            # 1. Topologically sort all nodes to be inserted to ensure the correctness of the insertion order.

            print('nodes_to_change',nodes_to_change)
            sorted_nodes_to_insert = self._topological_sort(nodes_to_change)
            if DEBUG: # Add DEBUG check
                print(f"  Global topological sort complete, {len(sorted_nodes_to_insert)} nodes in total.")
        except RuntimeError as e:
            if DEBUG: # Add DEBUG check
                print(f"  Error: {e}")
            return
        return sorted_nodes_to_insert ,activation_node_names

    def _modify_forward_graph(self, target_output_names: List[str], activations_to_recompute_name): # Keep original List[str] type hint
        """
        Completely resets the forward graph's output based on a given list of names.

        This method points the forward graph's `output` node to the nodes specified in `target_output_names`.
        This means only the values of these nodes will be preserved at the end of the forward pass.
        Afterwards, it utilizes `torch.fx`'s built-in functionality for dead code elimination,
        automatically removing intermediate computation nodes that no longer contribute to the
        final output, thus achieving memory optimization.

        Args:
            target_output_names (List[str]): A list of strings containing the names of the new
                                             output nodes for the forward graph. These nodes will
                                             become the final result of the forward graph's execution.
        """
        if DEBUG: # Add DEBUG check
            print(f"--- 2. Start resetting the forward graph's output based on the given list ---")
            
        # Step 1: Find the old output node to insert the new one before it, and finally delete it.
        fw_output_node = next((n for n in reversed(self.fw_gm.graph.nodes) if n.op == 'output'), None)
        if not fw_output_node:
            if DEBUG: # Add DEBUG check
                print("Warning: Forward graph has no output node, skipping modification.")
            return

        # Step 2: Convert the incoming target output [name list] to a [Node object list].
        target_nodes = []
        for name in target_output_names:
            if name in self.fw_name_to_node:
                target_nodes.append(self.fw_name_to_node[name])
            else:
                if DEBUG: # Add DEBUG check
                    print(f"Warning: Target output '{name}' does not exist in the forward graph and will be ignored.")

        final_outputs = list(dict.fromkeys(target_nodes))
        # 3. Atomically update the graph: create a new output, delete the old one.
        # new_output_node = None # The new_output_node variable is not used here, can be omitted
        fw_output_node.args=(tuple(final_outputs),)

        if DEBUG: # Add DEBUG check
            print(f"Forward graph output has been reset. The new output contains {len(final_outputs)} nodes: {[n.name for n in final_outputs]}")
        # ==========================================================
        # Step B: Use torch.fx built-in function to remove unused nodes (final optimized version)
        # ==========================================================
        if DEBUG: # Add DEBUG check
            print(f"--- 3. Start cleaning up unused nodes in the forward graph using the built-in API ---")

        # We can't use this, it will delete everything not related to the output, including `hccl_send`. 1. Call the built-in dead code elimination function.
        # self.fw_gm.graph.eliminate_dead_code()
        # ==========================================================
        # Step B: Manually prune dead code based on specified logic.
        # Traverse upwards from 'activations_to_recompute_name' and delete nodes with 0 users.
        # ==========================================================
        if DEBUG:
            print(f"--- 3. Start manually cleaning up unused nodes in the graph ---")

        # Use a deque as a worklist, starting from the activation points that need recomputation.
        worklist = deque()
        for node_name in activations_to_recompute_name:
            if node_name in self.fw_name_to_node:
                worklist.append(self.fw_name_to_node[node_name])

        deleted_node_count = 0
        # Use a set to avoid processing the same node multiple times for efficiency.
        visited = set()

        while worklist:
            node = worklist.popleft()
            if node in visited:
                continue
            visited.add(node)

            # If a node has no users (i.e., its output is not used) and it is not the graph's final output node,
            # we consider it "dead". The operation type check prevents deleting critical input/output nodes.
            if len(node.users) == 0 and node.op not in {'output', 'placeholder'}:
                if DEBUG:
                    print(f"Deleting node '{node.name}' (op: {node.op}) because it has no users.")

                # Before deleting the current node, add all its input nodes to the worklist.
                # This is because after the current node is deleted, its input nodes might also become dead code.
                for input_node in node.all_input_nodes:
                    if input_node not in visited:
                        worklist.append(input_node)

                # Erase the node from the graph.
                self.fw_gm.graph.erase_node(node)
                deleted_node_count += 1

        if DEBUG:
            print(f"Manual cleanup complete. Removed {deleted_node_count} unused nodes.")

    def _modify_backward_graph(self, activations_to_replace: List[str], sorted_nodes_to_insert: List[Node]):
        """
        修改此函数的签名，第二个参数现在是 List[Node]
        """
        # 2. Rename all placeholders to be replaced at once...
        placeholders_to_replace = {}
        for act_name in activations_to_replace:
            placeholder = self.bw_name_to_node.get(act_name)
            if placeholder and placeholder.users:
                placeholders_to_replace[act_name] = placeholder
                placeholder.name = f"_{act_name}_old_placeholder"
        self.re_init_maps()
        if DEBUG:
            print(f"   Batch renaming of {len(placeholders_to_replace)} old placeholders complete.")
        if not placeholders_to_replace:
            return

        # 3. Single traversal to dynamically insert all new nodes...
        # global_node_map 现在会映射【原始前向图节点】-> 【新建的、重命名后的反向图节点】
        global_node_map: Dict[fx.Node, fx.Node] = {} 
        
        # sorted_nodes_to_insert 现在是来自【原始前向图】的节点列表
        for node_to_create in sorted_nodes_to_insert:
            if DEBUG:
                print(f"\n   [Loop] Processing original forward node: {node_to_create.op} '{node_to_create.name}'")

            # =================================================================================
            # 核心修改点 1: 在这里直接进行重命名
            new_name = node_to_create.name + '_fw'
            # =================================================================================

            # Naming conflict check:
            # 这个检查现在变得很重要，确保新名字不会意外冲突
            if new_name in self.bw_name_to_node:
                # 允许的冲突是与我们即将替换的旧占位符的原名相同，但那些已经被重命名了
                # 所以任何冲突都是有问题的
                if not any(new_name == p.name for p in placeholders_to_replace.values()):
                    raise RuntimeError(
                        f"Naming conflict! Node '{new_name}' already exists in the backward graph and is not a placeholder to be replaced."
                    )
            
            # ... (lookup_fn_permissive 和参数解析部分保持不变) ...
            def lookup_fn_permissive(n: fx.Node) -> Optional[fx.Node]:
                if n in global_node_map:
                    return global_node_map[n]
                return self.bw_name_to_node.get(n.name)
            
            new_args = fx.node.map_arg(node_to_create.args, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)
            new_kwargs = fx.node.map_arg(node_to_create.kwargs, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)
            
            # ... (插入点定位逻辑保持不变) ...
            insertion_point = None
            all_input_nodes = []
            def get_input_nodes(a):
                if isinstance(a, fx.Node):
                    all_input_nodes.append(a)
            fx.node.map_arg(new_args, get_input_nodes)
            fx.node.map_arg(new_kwargs, get_input_nodes)
            
            latest_dependency_node = None
            for node in self.bw_gm.graph.nodes:
                if node in all_input_nodes:
                    latest_dependency_node = node
            
            if latest_dependency_node:
                insertion_point = latest_dependency_node
            else:
                last_placeholder = None
                for node in self.bw_gm.graph.nodes:
                    if node.op == 'placeholder':
                        last_placeholder = node
                insertion_point = last_placeholder
            
            if insertion_point is None:
                raise RuntimeError("Could not find any valid insertion point for the node!")

            with self.bw_gm.graph.inserting_after(insertion_point):
                meta_copy = copy.copy(node_to_create.meta)
                new_node = self.bw_gm.graph.create_node(
                    name=new_name,  # <--- 核心修改点 2: 使用新的名字
                    op=node_to_create.op, 
                    target=node_to_create.target,
                    args=new_args, 
                    kwargs=new_kwargs
                )
                new_node.meta = meta_copy
                if DEBUG:
                    print(f"   > Successfully created and inserted new node: '{new_node.name}'")

                # Update the maps.
                # 这里的映射关系是正确的：【原始节点】->【新创建的节点】
                global_node_map[node_to_create] = new_node
                self.bw_name_to_node[new_node.name] = new_node

        # 4. Replace all old placeholders...
        # 这部分逻辑现在可以正常工作了，因为 global_node_map 是正确的
        replaced_count = 0
        for act_name, old_placeholder_node in placeholders_to_replace.items():
            original_fw_node = self.fw_name_to_node.get(act_name)
            final_recomputed_node = global_node_map.get(original_fw_node)
            
            if final_recomputed_node:
                old_placeholder_node.replace_all_uses_with(final_recomputed_node)
                self.bw_gm.graph.erase_node(old_placeholder_node)
                replaced_count += 1
            else:
                if DEBUG:
                    print(f"   [Critical Error] Failed to find the final replacement node for '{act_name}'.")
        
        if DEBUG:
            print(f"   Successfully replaced {replaced_count} / {len(placeholders_to_replace)} old placeholders.")

        self.re_init_maps()
    def _topological_sort(self, nodes_to_sort: Set[fx.Node]) -> List[fx.Node]:
        """
        Performs a topological sort on a given set of `fx.Node`s.

        This is a crucial helper function to ensure that during the graph modification process,
        all nodes to be inserted are processed according to their dependencies (i.e., inputs
        are computed before the nodes that use them), thus avoiding errors due to
        unsatisfied dependencies.

        Args:
            nodes_to_sort (Set[fx.Node]): The set of nodes to be topologically sorted.

        Returns:
            List[fx.Node]: A list of nodes arranged in topological order.

        Raises:
            RuntimeError: If a cyclic dependency is detected in the node subgraph,
                          a topological sort cannot be performed.
        """
        if DEBUG: # Add DEBUG check
            print(f"[DEBUG] Starting topological sort for {len(nodes_to_sort)} nodes.")

        sorted_nodes = []
        in_degree = {node: 0 for node in nodes_to_sort}
        users_map = {node: [] for node in nodes_to_sort}

        # Traverse nodes to build in-degree and adjacency list (users_map)
        for u in nodes_to_sort:
            for v in u.users: # Iterate over all users of the current node `u`
                if v in nodes_to_sort: # If the user `v` is also in the set to be sorted
                    in_degree[v] += 1 # Increment the in-degree of user `v`
                    users_map[u].append(v) # Add `v` to the adjacency list of `u`

        # Initialize the queue with all nodes having an in-degree of 0 (i.e., nodes with no preceding dependencies)
        queue = deque([node for node in nodes_to_sort if in_degree[node] == 0])
        
        # Execute the BFS process for topological sorting
        while queue:
            u = queue.popleft() # Dequeue a node from the head of the queue
            sorted_nodes.append(u) # Add it to the sorted result
            for v in users_map[u]: # Iterate over the subsequent nodes that the current node `u` points to
                in_degree[v] -= 1 # Decrement the in-degree of the subsequent node `v`
                if in_degree[v] == 0: # If the in-degree of `v` becomes 0, it means all its dependencies have been processed
                    queue.append(v) # Enqueue `v`

        # Check if all nodes were sorted (i.e., whether a cycle exists)
        if len(sorted_nodes) != len(nodes_to_sort):
            if DEBUG: # Add DEBUG check
                print(f"[DEBUG] Error: Topological sort failed, cycle detected. Nodes to sort: {len(nodes_to_sort)}, nodes sorted: {len(sorted_nodes)}")
            raise RuntimeError("A cycle exists in the subgraph of nodes to be inserted, topological sort is not possible!")
            
        if DEBUG: # Add DEBUG check
            print(f"[DEBUG] Topological sort complete. Sorted result (partial): {[n.name for n in sorted_nodes[:min(5, len(sorted_nodes))]]}...")
        return sorted_nodes


    def rebuild_graph(self,gm: fx.GraphModule) -> fx.GraphModule:
        """
        接收一个 GraphModule，对其节点进行拓扑排序，然后重建一个新的、
        拓扑有序的 GraphModule。这是修复图顺序错误的稳健方法。

        Args:
            gm: 一个可能拓扑顺序混乱的 fx.GraphModule。

        Returns:
            一个全新的、保证拓扑有序的 fx.GraphModule。
        """
        
        # 1. 对输入图的所有节点进行拓扑排序
        # 我们通过手动构建邻接表和入度来进行排序，
        # 这种方法不依赖于可能已损坏的 .users 属性，因此非常稳健。
        all_nodes = list(gm.graph.nodes)
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        adj = [[] for _ in all_nodes]
        in_degree = [0] * len(all_nodes)
        
        for i, node in enumerate(all_nodes):
            for dep_node in node.all_input_nodes:
                if dep_node in node_map: # 确保依赖节点是图的一部分
                    dep_idx = node_map[dep_node]
                    adj[dep_idx].append(i) # 添加一条从依赖项到当前节点的边
                    in_degree[i] += 1

        # Kahn 算法实现
        queue = deque([i for i, deg in enumerate(in_degree) if deg == 0])
        sorted_nodes = []
        while queue:
            u_idx = queue.popleft()
            sorted_nodes.append(all_nodes[u_idx])
            for v_idx in adj[u_idx]:
                in_degree[v_idx] -= 1
                if in_degree[v_idx] == 0:
                    queue.append(v_idx)

        # 环路检测
        if len(sorted_nodes) != len(all_nodes):
            raise RuntimeError("Cycle detected in the graph, cannot topologically sort!")

        # 2. 创建新图，并使用 node_copy 和 arg_transform 重建
        new_graph = fx.Graph()
        old_to_new_map: Dict[Node, Node] = {}

        # 定义一个转换函数，用于将旧图中的节点参数映射到新图中的对应节点
        def arg_transform(n: Node) -> Node:
            return old_to_new_map[n]

        for old_node in sorted_nodes:
            # node_copy 会自动复制 op, target, name, meta 等所有属性。
            # arg_transform 则负责在复制过程中，动态地修复节点间的连接关系。
            new_node = new_graph.node_copy(old_node, arg_transform)
            old_to_new_map[old_node] = new_node
            
        # 3. 用新的、有序的图创建新的 GraphModule
        # 关键：第一个参数传入原始的 gm，这样可以保留对原始 nn.Module 的引用
        new_gm = fx.GraphModule(gm, new_graph)
        
        return new_gm