from typing import List, Any

from .. import logger


def get_activation_layer_ranks(fw_gm: Any, bw_gm: Any) -> List[int]:
    """
    通过数据流传播算法（Dataflow Propagation）为所有节点分配 layer_Rank，
    并收集跨越前向和反向的激活所在的层级，完美兼容残差和多分支结构。
    
    Args:
        fw_gm: 前向图的 GraphModule
        bw_gm: 反向图的 GraphModule
    
    Returns:
        List[int]: 排序后的唯一层级列表。
    """
    # 用于记录每个节点所属的 rank
    node_to_rank = {}

    # 1. 顺着图的拓扑顺序遍历所有节点，进行层级传播
    for node in fw_gm.graph.nodes:
        # 场景 A: 遇到我们的源头打标算子
        if node.op == "call_function" and str(node.target) == "my_compiler.mark_layer.default":
            rank = node.args[1]
            node_to_rank[node] = rank
            node.kwargs = {**node.kwargs, "layer_Rank": rank}
            continue

        # 场景 B: 普通算子，向其所有的父节点“问路”
        parent_ranks = []
        for input_node in node.all_input_nodes:
            if input_node in node_to_rank:
                parent_ranks.append(node_to_rank[input_node])

        # 核心：解决残差、多分支合并带来的冲突
        if parent_ranks:
            # 继承所有父节点中最深的层级 (Max Rank)
            inherited_rank = max(parent_ranks)
            node_to_rank[node] = inherited_rank
            node.kwargs = {**node.kwargs, "layer_Rank": inherited_rank}

    # 2. 收集跨越前向和反向的激活值
    output_node = next((n for n in reversed(fw_gm.graph.nodes) if n.op == "output"), None)
    if not output_node:
        return []

    bw_placeholder_names = {n.name for n in bw_gm.graph.nodes if n.op == "placeholder"}
    layer_ranks = set()

    for input_node in output_node.all_input_nodes:
        if input_node.name in bw_placeholder_names:
            rank = input_node.kwargs.get("layer_Rank")
            if rank is not None:
                layer_ranks.add(rank)

    sorted_ranks = sorted(layer_ranks)
    logger.info(f"[RecomputePass] 通过数据流推导成功提取 ranks: {sorted_ranks}")
    return sorted_ranks