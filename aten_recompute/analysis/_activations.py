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
from ..utils.graph_utils import get_fw_bw_boundary_info, get_output_node

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
    """
    output_node = get_output_node(fw_gm)
    if output_node is None:
        return {
            'activation_bytes': 0, 'primal_bytes': 0, 'total_bytes': 0,
            'num_activations': 0, 'num_primals': 0, 'skipped': 0,
            'activation_details': [], 'primal_details': [],
        }

    boundary = get_fw_bw_boundary_info(fw_gm, bw_gm)
    act_details: List[Dict] = []
    primal_details: List[Dict] = []
    skipped = 0

    for entry in boundary['saved_nodes']:
        detail = {
            'name': entry['name'],
            'shape': entry['shape'],
            'dtype': entry['dtype'],
            'bytes': entry['bytes'],
        }
        if not entry['is_tensor']:
            skipped += 1
            logger.debug(
                "[MemoryAnalyzer] 节点 '%s' meta['val']/tensor_meta 缺失或非 Tensor，跳过。",
                entry['name'],
            )
            continue
        if entry['kind'] == 'primal':
            primal_details.append(detail)
        else:
            act_details.append(detail)

    if skipped:
        logger.debug(
            "[MemoryAnalyzer] %d 个节点（标量 / SymInt）无法估算，已跳过。",
            skipped,
        )

    return {
        'activation_bytes': boundary['activation_bytes'],
        'primal_bytes': boundary['primal_bytes'],
        'total_bytes': boundary['total_bytes'],
        'num_activations': len(act_details),
        'num_primals': len(primal_details),
        'skipped': skipped,
        'activation_details': act_details,
        'primal_details': primal_details,
    }
