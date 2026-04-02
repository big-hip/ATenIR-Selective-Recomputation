"""
graph_utils.py — 共享的 FX 图工具函数

将多处重复出现的图遍历模式抽取为带完整文档的公共工具，
避免在 recompute.py / Recom_pass.py / memory_analysis.py 等处各自重写。
"""
from typing import Any, Dict, List, Optional, Set

import torch
import torch.fx as fx


__all__ = [
    "get_output_node",
    "get_saved_activations",
    "get_bw_placeholder_names",
    "get_fw_bw_boundary_info",
]


def get_output_node(gm: fx.GraphModule) -> Optional[fx.Node]:
    """
    返回图中最后一个 output 节点，不存在时返回 None。

    AOT Autograd 保证每张图恰好有一个 output 节点，位于节点列表末尾。
    从尾部反向遍历比从头遍历更快。
    """
    return next((n for n in reversed(gm.graph.nodes) if n.op == 'output'), None)


def get_bw_placeholder_names(bw_gm: fx.GraphModule) -> Set[str]:
    """返回 BW 图中非 tangent placeholder 的名字集合。"""
    return {
        n.name
        for n in bw_gm.graph.nodes
        if n.op == 'placeholder' and not n.name.startswith('tangents_')
    }


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
    bw_ph_names = get_bw_placeholder_names(bw_gm)
    output_node = get_output_node(fw_gm)
    if output_node is None:
        return set()
    return {n for n in output_node.all_input_nodes if n.name in bw_ph_names}


def _node_target_name(node: fx.Node) -> str:
    if node.op != 'call_function':
        return str(node.target)
    tgt = node.target
    if hasattr(tgt, '_opname'):
        return f"aten.{tgt._opname}"
    return str(tgt)


def _shape_list(shape: Any) -> Optional[List[int]]:
    try:
        return [int(dim) for dim in shape]
    except (TypeError, RuntimeError, AttributeError):
        return None


def _tensor_bytes_from_meta(val: Any, tensor_meta: Any) -> int:
    if isinstance(val, torch.Tensor):
        try:
            return int(val.numel()) * val.element_size()
        except (TypeError, RuntimeError):
            return 0
    if tensor_meta is not None:
        try:
            numel = 1
            for dim in tensor_meta.shape:
                numel *= int(dim)
            dtype = getattr(tensor_meta, 'dtype', None)
            itemsize = 0 if dtype is None else dtype.itemsize
            return numel * itemsize
        except (TypeError, AttributeError, RuntimeError):
            return 0
    return 0


def get_fw_bw_boundary_info(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> Dict:
    """
    返回 FW→BW 边界的统一摘要，用于图对照和分析复用。

    目前仅依赖 FX 结构与 meta['val'] / tensor_meta，不引入运行时执行。
    """
    fw_placeholder_names = {n.name for n in fw_gm.graph.nodes if n.op == 'placeholder'}
    bw_placeholder_names = get_bw_placeholder_names(bw_gm)
    bw_placeholder_order = [
        n.name
        for n in bw_gm.graph.nodes
        if n.op == 'placeholder' and not n.name.startswith('tangents_')
    ]
    output_node = get_output_node(fw_gm)
    saved_nodes = []
    fw_output_names = []
    output_index_by_name = {}

    if output_node is not None:
        fw_output_names = [node.name for node in output_node.all_input_nodes]
        output_index_by_name = {
            name: idx for idx, name in enumerate(fw_output_names)
        }
        for node in output_node.all_input_nodes:
            if node.name not in bw_placeholder_names:
                continue
            val = node.meta.get('val')
            tensor_meta = node.meta.get('tensor_meta')
            shape = None
            dtype = None
            if isinstance(val, torch.Tensor):
                shape = _shape_list(val.shape)
                dtype = str(val.dtype)
            elif tensor_meta is not None:
                shape = _shape_list(getattr(tensor_meta, 'shape', None))
                dtype = str(getattr(tensor_meta, 'dtype', None))

            entry = {
                'name': node.name,
                'op': node.op,
                'target': _node_target_name(node),
                'kind': 'primal' if node.name in fw_placeholder_names else 'activation',
                'is_tensor': isinstance(val, torch.Tensor) or tensor_meta is not None,
                'shape': shape,
                'dtype': dtype,
                'bytes': _tensor_bytes_from_meta(val, tensor_meta),
                'fw_output_index': output_index_by_name.get(node.name),
                'bw_placeholder_index': bw_placeholder_order.index(node.name),
            }
            saved_nodes.append(entry)

    activation_bytes = sum(n['bytes'] for n in saved_nodes if n['kind'] == 'activation')
    primal_bytes = sum(n['bytes'] for n in saved_nodes if n['kind'] == 'primal')

    return {
        'saved_nodes': saved_nodes,
        'saved_names': [n['name'] for n in saved_nodes],
        'activation_names': [n['name'] for n in saved_nodes if n['kind'] == 'activation'],
        'primal_names': [n['name'] for n in saved_nodes if n['kind'] == 'primal'],
        'activation_bytes': activation_bytes,
        'primal_bytes': primal_bytes,
        'total_bytes': activation_bytes + primal_bytes,
        'fw_output_order': fw_output_names,
        'bw_placeholder_order': bw_placeholder_order,
        'fw_to_bw_map': [
            {
                'name': n['name'],
                'fw_output_index': n['fw_output_index'],
                'bw_placeholder_index': n['bw_placeholder_index'],
            }
            for n in saved_nodes
        ],
    }


def summarize_graph_structure(gm: fx.GraphModule) -> Dict:
    nodes = list(gm.graph.nodes)
    node_names = [node.name for node in nodes]
    targets = [_node_target_name(node) for node in nodes]
    edges = []
    for node in nodes:
        for inp in node.all_input_nodes:
            edges.append((inp.name, node.name))

    return {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'node_names': node_names,
        'targets': targets,
        'target_sequence': [
            target for node, target in zip(nodes, targets)
            if node.op == 'call_function'
        ],
        'op_sequence': [node.op for node in nodes],
        'edge_sample': edges[:32],
    }


def compare_graph_structure(meta_gm: fx.GraphModule, runtime_gm: fx.GraphModule) -> Dict:
    meta_summary = summarize_graph_structure(meta_gm)
    runtime_summary = summarize_graph_structure(runtime_gm)

    meta_targets = meta_summary['target_sequence']
    runtime_targets = runtime_summary['target_sequence']
    prefix_len = 0
    for meta_t, runtime_t in zip(meta_targets, runtime_targets):
        if meta_t != runtime_t:
            break
        prefix_len += 1

    return {
        'meta_nodes': meta_summary['num_nodes'],
        'runtime_nodes': runtime_summary['num_nodes'],
        'meta_edges': meta_summary['num_edges'],
        'runtime_edges': runtime_summary['num_edges'],
        'same_node_sequence': meta_summary['node_names'] == runtime_summary['node_names'],
        'same_target_sequence': meta_summary['targets'] == runtime_summary['targets'],
        'target_prefix_match': prefix_len,
        'meta_summary': meta_summary,
        'runtime_summary': runtime_summary,
    }


def compare_boundary_info(meta_boundary: Dict, runtime_boundary: Dict) -> Dict:
    meta_saved = meta_boundary['saved_names']
    runtime_saved = runtime_boundary['saved_names']

    meta_activation = meta_boundary['activation_names']
    runtime_activation = runtime_boundary['activation_names']

    return {
        'same_saved_names': meta_saved == runtime_saved,
        'same_activation_names': meta_activation == runtime_activation,
        'same_primal_names': meta_boundary['primal_names'] == runtime_boundary['primal_names'],
        'same_fw_to_bw_map': meta_boundary['fw_to_bw_map'] == runtime_boundary['fw_to_bw_map'],
        'meta_saved_count': len(meta_saved),
        'runtime_saved_count': len(runtime_saved),
        'meta_activation_bytes': meta_boundary['activation_bytes'],
        'runtime_activation_bytes': runtime_boundary['activation_bytes'],
        'meta_primal_bytes': meta_boundary['primal_bytes'],
        'runtime_primal_bytes': runtime_boundary['primal_bytes'],
        'meta_saved_names': meta_saved,
        'runtime_saved_names': runtime_saved,
    }


__all__.extend([
    'summarize_graph_structure',
    'compare_graph_structure',
    'compare_boundary_info',
])
