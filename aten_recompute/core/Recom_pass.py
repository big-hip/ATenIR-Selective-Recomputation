"""
Recom_pass.py

RecomputePass：根据 RECOMPUTE 环境变量选择重计算策略，调用
ActivationRecomputation 执行图变换。

环境变量格式（JSON）:
    export RECOMPUTE='{"1": null}'                               # 全部重计算
    export RECOMPUTE='{"2": ["%relu"]}'                          # 按名称关键字
    export RECOMPUTE='{"3": [0, 2]}'                             # 按步幅选层
    export RECOMPUTE='{"4": 0.5}'                                # 按比例选前 50%
    export RECOMPUTE='{"5": ["relu.default", "dropout.default"]}' # 按算子类型
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch.fx as fx

from .recompute import ActivationRecomputation
from .get_activation_layer_ranks import get_activation_layer_ranks
from ..utils.logger import get_logger
from ..utils.graph_utils import get_output_node

logger = get_logger(__name__)


class RecomputePass:
    """
    根据 RECOMPUTE 环境变量选择重计算策略，对 FW/BW GraphModule 执行激活重计算 Pass。

    典型用法::

        pass_ = RecomputePass({"FW": fw_gm, "BW": bw_gm})
        fw_gm_opt, bw_gm_opt = pass_.run()
    """

    def __init__(self, graph_dict: Dict[str, fx.GraphModule]):
        """
        Parameters
        ----------
        graph_dict : {"FW": fw_gm, "BW": bw_gm}
            包含前向和反向 GraphModule 的字典。键名必须为 "FW" 和 "BW"。
        """
        self.fw_gm: fx.GraphModule = graph_dict["FW"]
        self.bw_gm: fx.GraphModule = graph_dict["BW"]

        # 由 _get_activation_layer_ranks() 填充
        self.sorted_ranks: List[int] = []
        self.node_name_to_rank: Dict[str, int] = {}

        # 从环境变量解析策略配置
        self.recompute_config: Dict[str, Any] = self._parse_recompute_env(
            os.getenv("RECOMPUTE", "{}")
        )
        if self.recompute_config:
            self.option, self.option_param = next(iter(self.recompute_config.items()))
        else:
            self.option, self.option_param = "0", None   # 默认：不重计算

        self.strategy_map: Dict[str, Callable[[Any], None]] = {
            "1": self._recompute_all,
            "2": self._recompute_by_list,
            "3": self._recompute_by_stride,
            "4": self._recompute_by_ratio,
            "5": self._recompute_by_op_type,
        }

    # ── 环境变量解析 ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_recompute_env(env_str: str) -> Dict[str, Any]:
        """
        解析 RECOMPUTE 环境变量。

        无效 JSON 或非 dict 时发出警告并回退到空配置（等价于策略 0：不重计算）。
        """
        try:
            parsed = json.loads(env_str)
            if not isinstance(parsed, dict):
                raise ValueError("RECOMPUTE 值必须是 JSON object（{...}）")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "[RecomputePass] RECOMPUTE 环境变量配置无效，回退到不重计算。原因: %s", e
            )
            return {}

    # ── 层级信息提取 ──────────────────────────────────────────────────────────

    def _get_activation_layer_ranks(self) -> List[int]:
        """
        调用 get_activation_layer_ranks 推导 FW→BW 边界上的层级列表，
        同时缓存 node_name_to_rank 供 _get_node_names_by_layer_ranks 查询。
        """
        self.sorted_ranks, self.node_name_to_rank = get_activation_layer_ranks(
            self.fw_gm, self.bw_gm
        )
        return self.sorted_ranks

    def _get_node_names_by_layer_ranks(self, target_ranks: set) -> List[str]:
        """
        返回 FW→BW 边界上属于指定 layer_rank 的节点名列表。

        使用 node_name_to_rank 字典（由 _get_activation_layer_ranks 填充）
        而非读取节点 kwargs，避免修改图节点的副作用。
        """
        output_node = get_output_node(self.fw_gm)
        if output_node is None:
            logger.warning("[RecomputePass] FW 图缺少 output 节点，无法选取激活。")
            return []

        bw_ph_names = {
            n.name for n in self.bw_gm.graph.nodes if n.op == 'placeholder'
        }
        matched = []
        for n in output_node.all_input_nodes:
            if n.name in bw_ph_names:
                rank = self.node_name_to_rank.get(n.name)
                if rank in target_ranks:
                    matched.append(n.name)

        logger.info(
            "[RecomputePass] 从 layers %s 中选取 %d 个激活节点。",
            sorted(target_ranks), len(matched),
        )
        return matched

    # ── 统一执行入口 ──────────────────────────────────────────────────────────

    def _run_recomputation(self, node_names: List[str]) -> None:
        """
        执行 ActivationRecomputation，更新 self.fw_gm / self.bw_gm。
        若 node_names 为空则直接跳过。
        """
        if not node_names:
            logger.info("[RecomputePass] 无节点需要重计算，跳过。")
            return
        logger.info(
            "[RecomputePass] 开始重计算 %d 个激活节点: %s", len(node_names), node_names
        )
        self.fw_gm, self.bw_gm = ActivationRecomputation(self.fw_gm, self.bw_gm).run(node_names)

    def run(self) -> Tuple[fx.GraphModule, fx.GraphModule]:
        """
        根据 RECOMPUTE 环境变量执行对应策略，返回优化后的 (fw_gm, bw_gm)。
        """
        self._get_activation_layer_ranks()
        strategy = self.strategy_map.get(self.option, self._no_recompute)
        strategy(self.option_param)
        return self.fw_gm, self.bw_gm

    # ── 策略实现 ──────────────────────────────────────────────────────────────

    def _no_recompute(self, _: Any) -> None:
        """策略 0：不重计算（默认）。"""
        logger.info("[RecomputePass] 策略 0：不执行重计算。")

    def _recompute_all(self, _: None) -> None:
        """策略 1：重计算所有中间激活（最大内存节省，最高重计算开销）。"""
        logger.info("[RecomputePass] 策略 1：重计算全部激活。")
        target_ranks = set(self.sorted_ranks)
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        self._run_recomputation(node_names)

    def _recompute_by_list(self, node_name_keywords: List[str]) -> None:
        """
        策略 2：重计算名称包含指定关键字的节点。

        Parameters
        ----------
        node_name_keywords : 节点名子串列表，如 ["%relu", "%norm"]。
        """
        logger.info("[RecomputePass] 策略 2：按名称关键字重计算: %s", node_name_keywords)
        node_names = [
            n.name for n in self.fw_gm.graph.nodes
            if any(k in n.name for k in node_name_keywords)
        ]
        self._run_recomputation(node_names)

    def _recompute_by_stride(self, stride_config: List[int]) -> None:
        """
        策略 3：按起始层 + 步幅模式选取层进行重计算。

        Parameters
        ----------
        stride_config : [start, stride]，例如 [0, 2] 表示第 0、2、4... 个 rank。

        注意：选取的是 sorted_ranks 列表中索引为 start, start+stride, ... 的元素，
        而非 rank 值本身满足模运算，如果 rank 不连续则结果符合"均匀间隔抽样"语义。
        """
        if not stride_config or len(stride_config) != 2:
            logger.warning(
                "[RecomputePass] 策略 3 配置格式错误，期望 [start, stride]，收到: %s",
                stride_config,
            )
            return
        start, stride = stride_config
        logger.info("[RecomputePass] 策略 3：start=%d, stride=%d", start, stride)
        target_ranks = {
            rank
            for idx, rank in enumerate(self.sorted_ranks)
            if idx >= start and (idx - start) % stride == 0
        }
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        self._run_recomputation(node_names)

    def _recompute_by_ratio(self, ratio: float) -> None:
        """
        策略 4：重计算前 N% 的层（按 sorted_ranks 顺序）。

        Parameters
        ----------
        ratio : [0, 1] 内的浮点数，如 0.5 表示重计算前 50% 层。
        """
        num_layers = int(len(self.sorted_ranks) * ratio)
        target_ranks = set(self.sorted_ranks[:num_layers])
        logger.info(
            "[RecomputePass] 策略 4：重计算前 %d/%d 层 (ratio=%.2f)。",
            num_layers, len(self.sorted_ranks), ratio,
        )
        node_names = self._get_node_names_by_layer_ranks(target_ranks)
        self._run_recomputation(node_names)

    def _recompute_by_op_type(self, op_types: List[str]) -> None:
        """
        策略 5：重计算指定算子类型的节点。

        Parameters
        ----------
        op_types : 算子类型名列表，如 ["relu.default", "dropout.default"]。
        """
        logger.info("[RecomputePass] 策略 5：按算子类型重计算: %s", op_types)
        node_names = [
            n.name
            for n in self.fw_gm.graph.nodes
            if n.op == "call_function"
            and hasattr(n.target, "__name__")
            and n.target.__name__ in op_types
        ]
        self._run_recomputation(node_names)
