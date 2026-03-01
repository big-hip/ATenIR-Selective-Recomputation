"""
graph_utils.py — 共享的 FX 图工具函数

将多处重复出现的图遍历模式抽取为带完整文档的公共工具，
避免在 recompute.py / Recom_pass.py / memory_analysis.py 等处各自重写。
"""
from typing import Optional, Set

import torch.fx as fx


__all__ = [
    "get_output_node",
    "get_saved_activations",
]


def get_output_node(gm: fx.GraphModule) -> Optional[fx.Node]:
    """
    返回图中最后一个 output 节点，不存在时返回 None。

    AOT Autograd 保证每张图恰好有一个 output 节点，位于节点列表末尾。
    从尾部反向遍历比从头遍历更快。
    """
    return next((n for n in reversed(gm.graph.nodes) if n.op == 'output'), None)


def get_saved_activations(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> Set[fx.Node]:
    """
    返回 FW→BW 边界上被保存的 FW 节点集合（不含 tangents）。

    具体定义：FW output 节点的输入中，其名字出现在 BW non-tangent
    placeholder 名字集里的节点，即 AOT Autograd saved-tensors 机制
    下显式传递给 BW 图的张量。

    Parameters
    ----------
    fw_gm : 前向 GraphModule
    bw_gm : 反向 GraphModule

    Returns
    -------
    Set of fx.Node from fw_gm that cross the FW→BW boundary.
    空集合表示图为空或边界无保存张量。
    """
    bw_ph_names: Set[str] = {
        n.name
        for n in bw_gm.graph.nodes
        if n.op == 'placeholder' and not n.name.startswith('tangents_')
    }
    output_node = get_output_node(fw_gm)
    if output_node is None:
        return set()
    return {n for n in output_node.all_input_nodes if n.name in bw_ph_names}
