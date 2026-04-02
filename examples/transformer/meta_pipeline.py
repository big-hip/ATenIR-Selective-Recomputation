"""Shared meta-first capture helpers for transformer example workflows."""

from __future__ import annotations

import collections
from typing import Callable, Dict

import torch
import torch.nn as nn

from aten_recompute.core import CompilerBackend, inject_layer_tags
from aten_recompute.utils.graph_utils import (
    compare_boundary_info,
    compare_graph_structure,
    get_fw_bw_boundary_info,
)


def inject_transformer_layer_tags(transformer: nn.Module) -> None:
    """Inject rank tags into encoder/decoder layers in a stable order."""
    enc_layers = [(layer, i) for i, layer in enumerate(transformer.encoder_layers)]
    dec_layers = [
        (layer, len(transformer.encoder_layers) + i)
        for i, layer in enumerate(transformer.decoder_layers)
    ]
    inject_layer_tags(enc_layers + dec_layers)


def run_train_step(compiled_model, src_data, tgt_data, tgt_vocab_size: int, criterion) -> float:
    """Run one forward+backward step and return scalar loss."""
    output = compiled_model(src_data, tgt_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    return float(loss.item())


def capture_static_with_retry(
    model: nn.Module,
    sample_inputs: tuple,
    loss_builder: Callable,
    strategy_config: dict,
    *,
    dynamic: bool = True,
    use_meta: bool = True,
):
    """
    Capture static FW/BW graphs with decomp retry fallback.

    - First try `use_decomp=True`
    - If execution fails, retry with `use_decomp=False`
    """
    import torch._dynamo

    last_exc = None
    for use_decomp in (True, False):
        torch._dynamo.reset()
        backend = CompilerBackend(
            strategy_config=strategy_config,
            save_ir=False,
            mode="static",
            use_meta=use_meta,
            use_decomp=use_decomp,
        )
        compiled = torch.compile(model, backend=backend, dynamic=dynamic)
        try:
            output = compiled(*sample_inputs)
            loss = loss_builder(output)
            loss.backward()
            return backend, use_decomp, None
        except Exception as exc:  # pragma: no cover - runtime path only
            last_exc = exc
            if backend.fw_gm is not None and backend.bw_gm is not None:
                return backend, use_decomp, exc

    raise RuntimeError(f"static capture failed: {type(last_exc).__name__}: {last_exc}")


def graph_signature(gm) -> collections.Counter:
    """Build op-count signature from a GraphModule."""
    sig = collections.Counter()
    if gm is None:
        return sig
    for node in gm.graph.nodes:
        if node.op == "call_function":
            tgt = node.target
            if hasattr(tgt, "_opname"):
                name = f"aten.{tgt._opname}"
            else:
                name = str(tgt)
            sig[name] += 1
    return sig


def compare_graph_signatures(meta_gm, runtime_gm) -> Dict:
    """Compare two graph signatures and return compact metrics."""
    meta_sig = graph_signature(meta_gm)
    runtime_sig = graph_signature(runtime_gm)

    shared = set(meta_sig) & set(runtime_sig)
    all_keys = set(meta_sig) | set(runtime_sig)

    diffs = []
    for op in all_keys:
        delta = meta_sig.get(op, 0) - runtime_sig.get(op, 0)
        if delta != 0:
            diffs.append((abs(delta), delta, op))
    diffs.sort(reverse=True)

    meta_nodes = sum(meta_sig.values())
    runtime_nodes = sum(runtime_sig.values())

    return {
        "meta_nodes": meta_nodes,
        "runtime_nodes": runtime_nodes,
        "ratio": (meta_nodes / runtime_nodes) if runtime_nodes > 0 else 0.0,
        "overlap": (len(shared) / len(all_keys)) if all_keys else 1.0,
        "top_diffs": [{"op": op, "delta": delta} for _, delta, op in diffs[:5]],
    }



def compare_capture_semantics(meta_backend, runtime_backend) -> Dict:
    fw_sig = compare_graph_signatures(meta_backend.fw_gm, runtime_backend.fw_gm)
    bw_sig = compare_graph_signatures(meta_backend.bw_gm, runtime_backend.bw_gm)
    fw_struct = compare_graph_structure(meta_backend.fw_gm, runtime_backend.fw_gm)
    bw_struct = compare_graph_structure(meta_backend.bw_gm, runtime_backend.bw_gm)
    fw_boundary = compare_boundary_info(
        get_fw_bw_boundary_info(meta_backend.fw_gm, meta_backend.bw_gm),
        get_fw_bw_boundary_info(runtime_backend.fw_gm, runtime_backend.bw_gm),
    )
    return {
        "fw_signature": fw_sig,
        "bw_signature": bw_sig,
        "fw_structure": fw_struct,
        "bw_structure": bw_struct,
        "fw_bw_boundary": fw_boundary,
    }



def print_capture_semantics_report(report: Dict) -> None:
    fw_sig = report["fw_signature"]
    bw_sig = report["bw_signature"]
    fw_struct = report["fw_structure"]
    bw_struct = report["bw_structure"]
    boundary = report["fw_bw_boundary"]

    print(
        f"  [FW] meta/runtime 节点数: {fw_sig['meta_nodes']}/{fw_sig['runtime_nodes']} "
        f"(ratio={fw_sig['ratio']:.3f})"
    )
    print(f"  [FW] 算子集合重叠: {fw_sig['overlap']:.1%}")
    print(
        f"  [FW] target 序列一致: {fw_struct['same_target_sequence']} "
        f"(公共前缀 {fw_struct['target_prefix_match']})"
    )
    if fw_sig["top_diffs"]:
        print("  [FW] Top-5 算子计数差异 (meta - runtime):")
        for item in fw_sig["top_diffs"]:
            print(f"    {item['op']}: {item['delta']:+d}")

    print(
        f"  [BW] meta/runtime 节点数: {bw_sig['meta_nodes']}/{bw_sig['runtime_nodes']} "
        f"(ratio={bw_sig['ratio']:.3f})"
    )
    print(f"  [BW] 算子集合重叠: {bw_sig['overlap']:.1%}")
    print(
        f"  [BW] target 序列一致: {bw_struct['same_target_sequence']} "
        f"(公共前缀 {bw_struct['target_prefix_match']})"
    )
    if bw_sig["top_diffs"]:
        print("  [BW] Top-5 算子计数差异 (meta - runtime):")
        for item in bw_sig["top_diffs"]:
            print(f"    {item['op']}: {item['delta']:+d}")

    print(
        "  [Boundary] saved_names一致: "
        f"{boundary['same_saved_names']} "
        f"({boundary['meta_saved_count']}/{boundary['runtime_saved_count']})"
    )
    print(
        "  [Boundary] activation/primal 分类一致: "
        f"{boundary['same_activation_names'] and boundary['same_primal_names']}"
    )
    print(f"  [Boundary] FW→BW 映射一致: {boundary['same_fw_to_bw_map']}")
    print(
        f"  [Boundary] activation bytes(meta/runtime): "
        f"{boundary['meta_activation_bytes']}/{boundary['runtime_activation_bytes']}"
    )
    print(
        f"  [Boundary] primal bytes(meta/runtime): "
        f"{boundary['meta_primal_bytes']}/{boundary['runtime_primal_bytes']}"
    )

    if not boundary["same_saved_names"]:
        only_meta = [
            name for name in boundary['meta_saved_names']
            if name not in set(boundary['runtime_saved_names'])
        ]
        only_runtime = [
            name for name in boundary['runtime_saved_names']
            if name not in set(boundary['meta_saved_names'])
        ]
        if only_meta:
            print(f"  [Boundary] 仅 meta 保存: {only_meta[:8]}")
        if only_runtime:
            print(f"  [Boundary] 仅 runtime 保存: {only_runtime[:8]}")

    partition_equivalent = (
        fw_struct['same_target_sequence']
        and bw_struct['same_target_sequence']
        and boundary['same_saved_names']
        and boundary['same_fw_to_bw_map']
    )
    print(f"  [Conclusion] partition 相关语义等价: {partition_equivalent}")
    print(f"  [Conclusion] 仅 op-count 对照不足，已输出结构级/边界级对照。")

    report['partition_equivalent'] = partition_equivalent
    report['boundary_equivalent'] = (
        boundary['same_saved_names']
        and boundary['same_activation_names']
        and boundary['same_primal_names']
        and boundary['same_fw_to_bw_map']
    )
    report['structure_equivalent'] = (
        fw_struct['same_target_sequence'] and bw_struct['same_target_sequence']
    )
    return report


__all__ = [
    'inject_transformer_layer_tags',
    'run_train_step',
    'capture_static_with_retry',
    'graph_signature',
    'compare_graph_signatures',
    'compare_capture_semantics',
    'print_capture_semantics_report',
]
