import torch
import torch.fx as fx
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import copy
from torch.fx.node import Node
from .. import logger  # 引入你配置好的 logger 工具

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
        # 0
        true_activations_to_recompute_name = self._get_true_recomputation_candidates(activations_to_recompute_name)
        # 1. Identify the nodes to be changed (the subgraph leading to the recomputation targets); these should point to the output.
        sorted_nodes_to_insert, activation_node_names = self._identify_nodes_to_change(true_activations_to_recompute_name)

        # 3. Modify the backward graph.
        logger.debug(f"True activations to recompute: {true_activations_to_recompute_name}")
        logger.debug(f"Sorted nodes to insert: {sorted_nodes_to_insert}")

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

        logger.info("Activation Recomputation Pass executed successfully.")
        return self.fw_gm, self.bw_gm

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

    def _identify_nodes_to_change(self, activations_to_recompute: List[str]) -> Tuple[Set[fx.Node], Set[str]]:
        """
        Identifies all nodes in the forward graph that need to be recomputed.
        """
        logger.debug("--- 1. Start identifying nodes to change ---")
        
        output_node = None
        # Traverse the graph nodes from back to front to find the first 'output' node
        for node in reversed(self.fw_gm.graph.nodes):
            if node.op == 'output':
                output_node = node
                break

        nodes_to_change: Set[fx.Node] = set()
        activation_node_names = set()
        
        output_node = None # Re-find the output_node to ensure correctness
        for node in reversed(self.fw_gm.graph.nodes):
            if node.op == 'output':
                output_node = node
                break
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
            
            nodes_to_change.add(current_node)
            logger.debug(f"Processing node: {current_node}, inputs: {current_node.all_input_nodes}")
            
            if current_node.op != 'placeholder':
                for input_node in current_node.all_input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.append(input_node)
                        
        logger.debug(f"Identification complete, found {len(nodes_to_change)} nodes to be changed: {[n.name for n in sorted(list(nodes_to_change), key=lambda x: x.name)]}")
        
        activation_node_names.update(
            node.name for node in nodes_to_change if node.op == 'placeholder'
        )
        logger.debug(f"Identification complete, found {len(activation_node_names)} activation nodes: {list(activation_node_names)}")
        
        try:
            logger.debug(f"nodes_to_change: {nodes_to_change}")
            sorted_nodes_to_insert = self._topological_sort(nodes_to_change)
            logger.debug(f"Global topological sort complete, {len(sorted_nodes_to_insert)} nodes in total.")
        except RuntimeError as e:
            logger.error(f"Topological sort error: {e}")
            return set(), set()
            
        return sorted_nodes_to_insert, activation_node_names

    def _modify_forward_graph(self, target_output_names: List[str], activations_to_recompute_name):
        """
        Completely resets the forward graph's output based on a given list of names.
        """
        logger.debug("--- 2. Start resetting the forward graph's output based on the given list ---")
            
        fw_output_node = next((n for n in reversed(self.fw_gm.graph.nodes) if n.op == 'output'), None)
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
        
        for node_to_create in sorted_nodes_to_insert:
            logger.debug(f"[Loop] Processing original forward node: {node_to_create.op} '{node_to_create.name}'")

            new_name = node_to_create.name + '_fw'

            if new_name in self.bw_name_to_node:
                if not any(new_name == p.name for p in placeholders_to_replace.values()):
                    raise RuntimeError(
                        f"Naming conflict! Node '{new_name}' already exists in the backward graph and is not a placeholder to be replaced."
                    )
            
            def lookup_fn_permissive(n: fx.Node) -> Optional[fx.Node]:
                if n in global_node_map:
                    return global_node_map[n]
                return self.bw_name_to_node.get(n.name)
            
            new_args = fx.node.map_arg(node_to_create.args, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)
            new_kwargs = fx.node.map_arg(node_to_create.kwargs, lambda n: lookup_fn_permissive(n) if isinstance(n, fx.Node) else n)
            
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
                    name=new_name, 
                    op=node_to_create.op, 
                    target=node_to_create.target,
                    args=new_args, 
                    kwargs=new_kwargs
                )
                new_node.meta = meta_copy
                logger.debug(f"> Successfully created and inserted new node: '{new_node.name}'")

                global_node_map[node_to_create] = new_node
                self.bw_name_to_node[new_node.name] = new_node

        replaced_count = 0
        for act_name, old_placeholder_node in placeholders_to_replace.items():
            original_fw_node = self.fw_name_to_node.get(act_name)
            final_recomputed_node = global_node_map.get(original_fw_node)
            
            if final_recomputed_node:
                old_placeholder_node.replace_all_uses_with(final_recomputed_node)
                self.bw_gm.graph.erase_node(old_placeholder_node)
                replaced_count += 1
            else:
                logger.error(f"[Critical Error] Failed to find the final replacement node for '{act_name}'.")
        
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