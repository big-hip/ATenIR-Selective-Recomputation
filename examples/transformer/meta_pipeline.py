"""Shared meta-first capture helpers for transformer example workflows."""

from __future__ import annotations

import collections
from typing import Callable, Dict

import torch
import torch.nn as nn

from aten_recompute.core import CompilerBackend, inject_layer_tags


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
