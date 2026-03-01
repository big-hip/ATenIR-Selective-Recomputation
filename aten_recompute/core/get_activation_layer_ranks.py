"""
get_activation_layer_ranks.py

通过数据流传播算法（Dataflow Propagation）为 FW 图中的所有节点推导
所属 layer_rank，并收集跨越 FW→BW 边界的激活所在层级。

**设计原则**：本模块为纯分析函数，不修改任何图节点的 kwargs。
layer_rank 信息通过返回值 node_name_to_rank 对外传递，
由调用方（Recom_pass._get_node_names_by_layer_ranks）按需查询。
"""
from typing import Dict, List, Optional, Tuple, Any

from ..utils.graph_utils import get_output_node
from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_activation_layer_ranks(
    fw_gm: Any,
    bw_gm: Any,
) -> Tuple[List[int], Dict[str, int]]:
    """
    推导前向图中各节点的 layer_rank，并返回跨越 FW→BW 边界的层级信息。

    算法：按拓扑顺序遍历 FW 图，遇到 mark_layer 节点时记录其 rank，
    其余节点继承所有父节点中的最大 rank（Max-Rank 规则，兼容残差/多分支）。

    Parameters
    ----------
    fw_gm : 前向 GraphModule（含 mark_layer 节点）
    bw_gm : 反向 GraphModule（用于确定 FW→BW 边界）

    Returns
    -------
    sorted_ranks : List[int]
        出现在 FW→BW 边界上的 layer_rank 列表，已去重排序。
    node_name_to_rank : Dict[str, int]
        FW 图中每个节点名 → layer_rank 的映射，供调用方按名查询。
        未被任何 mark_layer 影响的节点不在此字典中。
    """
    node_to_rank: Dict = {}

    # ── 1. 按拓扑顺序传播 layer_rank ────────────────────────────────────────
    for node in fw_gm.graph.nodes:
        # mark_layer 节点：rank 来自 args[1]
        if (node.op == "call_function"
                and str(node.target) == "my_compiler.mark_layer.default"):
            rank = node.args[1]
            node_to_rank[node] = rank
            continue

        # 普通节点：继承所有父节点中最大的 rank
        parent_ranks = [
            node_to_rank[inp]
            for inp in node.all_input_nodes
            if inp in node_to_rank
        ]
        if parent_ranks:
            node_to_rank[node] = max(parent_ranks)

    # ── 2. 收集 FW→BW 边界上的激活层级 ─────────────────────────────────────
    output_node = get_output_node(fw_gm)
    if output_node is None:
        return [], {}

    bw_ph_names = {
        n.name for n in bw_gm.graph.nodes if n.op == "placeholder"
    }
    layer_ranks_at_boundary: set = set()

    for fw_out_node in output_node.all_input_nodes:
        if fw_out_node.name in bw_ph_names and fw_out_node in node_to_rank:
            layer_ranks_at_boundary.add(node_to_rank[fw_out_node])

    sorted_ranks = sorted(layer_ranks_at_boundary)

    # 构建 node_name → rank 映射，供调用方查询（避免修改图节点 kwargs）
    node_name_to_rank: Dict[str, int] = {
        n.name: r for n, r in node_to_rank.items()
    }

    logger.info(
        "[get_activation_layer_ranks] 通过数据流推导成功提取 ranks: %s", sorted_ranks
    )
    return sorted_ranks, node_name_to_rank
