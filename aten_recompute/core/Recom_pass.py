import os
import json
from typing import Any, Dict, List, Callable, Optional
from .recompute import ActivationRecomputation
from .. import logger
import copy

class RecomputePass:
    """
    A recomputation pass that applies different activation recomputation strategies
    based on the RECOMPUTE environment variable.

    The environment variable should be a JSON-formatted string, for example:
        export RECOMPUTE='{"1": null}'                          # Recompute all
        export RECOMPUTE='{"2": ["%relu"]}'                     # Recompute specific nodes
        export RECOMPUTE='{"3": [0, 2]}'                        # Recompute every 2 layers starting from 0
        export RECOMPUTE='{"4": 0.5}'                           # Recompute first 50% layers
        export RECOMPUTE='{"5": ["relu.default", "dropout.default"]}' # Recompute by op type

    Each key in the JSON corresponds to a strategy ID (as string),
    and its value is the parameter required for that strategy.
    """

    def __init__(self, dist_ir: Any):
        """
        Initialize the recomputation pass.

        Args:
            dist_ir (Any): The distributed intermediate representation (IR) graph
                           on which recomputation will be applied.
        """
        self.dist_ir =dist_ir
        self.fw_gm = self.dist_ir["FW"]
        self.bw_gm = self.dist_ir["BW"]
        
        # Load recompute config from environment
        self.recompute_config: Dict[str, Any] = self._parse_recompute_env(os.getenv("RECOMPUTE", "{}"))

        # Extract strategy option and its associated parameters
        if self.recompute_config:
            self.option, self.option_param = next(iter(self.recompute_config.items()))
        else:
            self.option, self.option_param = "0", None  # Default: no recomputation

        # Register all available strategies
        self.strategy_map: Dict[str, Callable[[Any], None]] = {
            "1": self._recompute_all,
            "2": self._recompute_by_list,
            "3": self._recompute_by_stride,
            "4": self._recompute_by_ratio,
            "5": self._recompute_by_op_type
        }

    def _parse_recompute_env(self, env_str: str) -> Dict[str, Any]:
        """
        Parse the RECOMPUTE environment variable.

        Args:
            env_str (str): JSON string from environment variable.

        Returns:
            Dict[str, Any]: Parsed strategy config as a dictionary.

        Example:
            '{"2": ["relu"]}' => {"2": ["relu"]}
        """
        try:
            parsed = json.loads(env_str)
            if not isinstance(parsed, dict):
                raise ValueError
            return parsed
        except (json.JSONDecodeError, ValueError):
            logger.info("[RecomputePass] Invalid RECOMPUTE config. Fallback to no recompute.")
            return {}
    

    def _get_activation_layer_ranks(self) -> List[int]:
        """
        Collect all unique layer_Rank values from activations.

        An activation is defined as a forward node that is an input to the
        output node, and also appears as a placeholder in the backward graph.

        Returns:
            List[int]: Sorted list of unique layer_Rank values.
        """
        output_node = next((n for n in reversed(self.fw_gm.graph.nodes) if n.op == 'output'), None)
        if output_node is None:
            logger.info("[RecomputePass] No output node found in forward graph.")
            return []

        # Collect all placeholder node names in the backward graph
        bw_placeholder_names = {n.name for n in self.bw_gm.graph.nodes if n.op == 'placeholder'}

        # Find matching nodes in output's inputs
        layer_ranks = set()
        for input_node in output_node.all_input_nodes:
            if input_node.name in bw_placeholder_names:
                layer_rank = input_node.kwargs.get("layer_Rank")
                if layer_rank is not None:
                    layer_ranks.add(layer_rank)

        self.sorted_ranks = sorted(layer_ranks)
        logger.info(f"[RecomputePass] Activation layer ranks: {self.sorted_ranks}")
        return self.sorted_ranks

    
    def _get_node_names_by_layer_ranks(self, target_ranks: set) -> List[str]:
        """
        Return all node names from the forward graph that match the given layer ranks.

        Args:
            target_ranks (set): Set of layer_Rank values.

        Returns:
            List[str]: Names of matched nodes.
        """
        output_node = next((n for n in reversed(self.fw_gm.graph.nodes) if n.op == 'output'), None)
        bw_placeholder_names = {n.name for n in self.bw_gm.graph.nodes if n.op == 'placeholder'}

        matched = []
        for input_node in output_node.all_input_nodes:
            if input_node.name in bw_placeholder_names:
                rank = input_node.kwargs.get("layer_Rank")
                if rank in target_ranks:
                    matched.append(input_node.name)
        logger.info(f"[RecomputePass] Selected {len(matched)} activations from layers {sorted(target_ranks)}")
        return matched


    def run(self) -> None:
        """
        Execute the recomputation strategy selected via RECOMPUTE.
        """
        self._get_activation_layer_ranks()
        if hasattr(self,'sorted_ranks'):
            strategy = self.strategy_map.get(self.option, self._no_recompute)
            strategy(self.option_param)

    def _recompute_all(self, _: None) -> None:
        """
        Strategy 1: Recompute all intermediate activations in the forward graph.
        This maximizes memory savings but adds the highest recomputation cost.
        """
        logger.info("[RecomputePass] Strategy 1: Recompute all activations.")
        target_ranks = set(self.sorted_ranks)
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        logger.info(f"[RecomputePass] Recomputing {len(node_names)} activation nodes: {node_names}")
        ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)

    def _recompute_by_list(self, node_name_keywords: List[str]) -> None:
        """
        Strategy 2: Recompute only nodes whose names match the given list.

        Args:
            node_name_keywords: List of node name substrings (e.g., ["%relu", "%norm"]).
        """
        logger.info(f"[RecomputePass] Strategy 2: Recompute nodes by name: {node_name_keywords}")
        node_names = [
            n.name for n in self.fw_gm.graph.nodes
            if any(k in n.name for k in node_name_keywords)
        ]
        logger.info(f"[RecomputePass] Recomputing {len(node_names)} activation nodes: {node_names}")
        ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)

    def _recompute_by_stride(self, stride_config: List[int]) -> None:
        """
        Strategy 3: Recompute layers following a start + stride * i pattern.

        Args:
            stride_config (List[int]): [start, stride], indicating which layers to recompute.
                                       For example, [0, 2] means layers 0, 2, 4, ...
        """
        if not stride_config or len(stride_config) != 2:
            logger.info("[RecomputePass] Invalid stride config. Expected format: [start, stride]")
            return
        start, stride = stride_config
        logger.info(f"[RecomputePass] Strategy 3: Recompute layers with start={start}, stride={stride}")
        target_ranks = {
            rank for idx, rank in enumerate(self.sorted_ranks)
            if (idx - start) % stride == 0 and idx >= start
        }
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        logger.info(f"[RecomputePass] Recomputing {len(node_names)} activation nodes: {node_names}")
        ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)

       
    def _recompute_by_ratio(self, ratio: float) -> None:
        """
        Strategy 4: Recompute the first N% of layers based on a given ratio.

        Args:
            ratio (float): Ratio in [0, 1], indicating how much of the early layers to recompute.
                           For example, 0.5 means recompute the first 50% of layers.
        """
        num_layers = int(len(self.sorted_ranks) * ratio)
        target_ranks = set(self.sorted_ranks[:num_layers])
        logger.info(f"[RecomputePass] Strategy 4: Recompute first {num_layers}/{len(self.sorted_ranks)} layers.")
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        logger.info(f"[RecomputePass] Recomputing {len(node_names)} activation nodes: {node_names}")
        ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)



    def _recompute_by_op_type(self, op_types: List[str]) -> None:
        """
        Strategy 5: Recompute only nodes of specific operator types.

        Args:
            op_types (List[str]): List of operator type strings (e.g., ["aten.relu", "aten.dropout"]).
        """
        logger.info(f"[RecomputePass] Strategy 5: Recompute nodes by op types: {op_types}")
        # Collect all node names in the forward graph matching given op types and having layer_Rank
        node_names = []
        for n in self.fw_gm.graph.nodes:
            if n.op == "call_function" and hasattr(n.target, "__module__"):
                target_name = n.target.__name__
                if target_name in op_types:
                    node_names.append(n.name)
        # node_names = [
        #     n.name for n in self.fw_gm.graph.nodes
        #     if n.op == "call_function" and
        #        hasattr(n.target, "__module__") and
        #        f"{n.target.__name__}" in op_types
        # ]

        logger.info(f"[RecomputePass] Recomputing {len(node_names)} activation nodes: {node_names}")
        ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)



    def _no_recompute(self, _: Any) -> None:
        """
        Default strategy: No recomputation is applied.
        """
        logger.info("[RecomputePass] Strategy 0: No recomputation applied.")
        return 