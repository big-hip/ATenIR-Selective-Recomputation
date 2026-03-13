"""
_activations.py

FW→BW 边界激活分析工具函数。
从 FX 图的 meta['val'] 中提取 saved activation 信息，
区分中间激活（重计算目标）和参数透传（始终常驻）。
"""

from typing import Dict, List

import torch
import torch.fx as fx

from ..utils.logger import get_logger
from ..utils.graph_utils import get_output_node

logger = get_logger(__name__)


def _fmt_bytes(n) -> str:
    """将字节数格式化为易读的 KB / MB / GB 字符串。
    兼容 SymInt / SymFloat：先用 float() 强制具化，若失败则返回占位符。
    """
    try:
        n = float(n)
    except (TypeError, RuntimeError):
        return "? B (symbolic)"
    if n == 0:
        return "0 B"
    for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if abs(n) >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return f"{int(n)} B"


def _tensor_bytes(val) -> int:
    """从 FakeTensor / 真实 Tensor 估算字节数；非 Tensor 或符号形状无法具化时返回 0。"""
    if val is None or not isinstance(val, torch.Tensor):
        return 0
    try:
        numel = int(val.numel())
    except (TypeError, RuntimeError):
        return 0
    return numel * val.element_size()


def _saved_activation_bytes(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> Dict:
    """
    统计 FW→BW 边界上保存的张量，并将其分类为：
      - activations : FW 中间计算结果（op != 'placeholder'），是重计算真正要消除的对象
      - primals     : FW placeholder 节点直接透传给 BW（模型参数），本就长驻显存，
                      重计算无法"节省"这部分，但添加新的 primal 会让此类增加

    Returns:
        {
            'activation_bytes':  int,
            'primal_bytes':      int,
            'total_bytes':       int,
            'num_activations':   int,
            'num_primals':       int,
            'skipped':           int,
            'activation_details': list,
            'primal_details':    list,
        }
    """
    fw_placeholder_names = {
        n.name for n in fw_gm.graph.nodes if n.op == 'placeholder'
    }

    bw_ph_names = {
        n.name
        for n in bw_gm.graph.nodes
        if n.op == 'placeholder' and not n.name.startswith('tangents_')
    }

    output_node = get_output_node(fw_gm)
    if output_node is None:
        return {
            'activation_bytes': 0, 'primal_bytes': 0, 'total_bytes': 0,
            'num_activations': 0, 'num_primals': 0, 'skipped': 0,
            'activation_details': [], 'primal_details': [],
        }

    act_details:    List[Dict] = []
    primal_details: List[Dict] = []
    act_bytes    = 0
    primal_bytes = 0
    skipped      = 0

    for node in output_node.all_input_nodes:
        if node.name not in bw_ph_names:
            continue
        val = node.meta.get('val')
        if val is None or not isinstance(val, torch.Tensor):
            skipped += 1
            logger.debug(
                "[MemoryAnalyzer] 节点 '%s' meta['val'] 缺失或非 Tensor，跳过。",
                node.name,
            )
            continue

        nb = _tensor_bytes(val)
        entry = {
            'name':  node.name,
            'shape': [int(d) for d in val.shape],
            'dtype': str(val.dtype),
            'bytes': nb,
        }

        if node.name in fw_placeholder_names:
            primal_bytes += nb
            primal_details.append(entry)
        else:
            act_bytes += nb
            act_details.append(entry)

    if skipped:
        logger.debug(
            "[MemoryAnalyzer] %d 个节点（标量 / SymInt）无法估算，已跳过。",
            skipped,
        )

    return {
        'activation_bytes':  act_bytes,
        'primal_bytes':      primal_bytes,
        'total_bytes':       act_bytes + primal_bytes,
        'num_activations':   len(act_details),
        'num_primals':       len(primal_details),
        'skipped':           skipped,
        'activation_details': act_details,
        'primal_details':    primal_details,
    }
