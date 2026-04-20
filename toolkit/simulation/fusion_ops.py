"""Operator classification for fusion-aware memory estimation (L2.5).

Inductor fuses consecutive pointwise/reduction ops into single Triton kernels,
eliminating intermediate tensor allocations.  This module classifies ATen IR ops
into *extern* (non-fusable, e.g. mm/bmm/sdpa) and *fusable* (pointwise,
reduction, elementwise) categories, mirroring the classification that Inductor's
Scheduler performs internally.

The EXTERN_OPS set is based on PyTorch 2.6.0 torch._inductor internals:
- ir.ExternKernel subclasses handle CUBLAS / cuDNN / Flash-Attention ops
- ir.ComputedBuffer (SchedulerNode) handles fusable pointwise / reduction ops
"""

import torch
import torch.fx as fx


# ── Extern ops: become ExternKernelSchedulerNode in Inductor ──────────
# These ops are dispatched to vendor libraries (CUBLAS, cuDNN, Flash-Attn)
# and cannot be fused with neighboring pointwise ops.

EXTERN_OPS: set = set()


def _populate_extern_ops() -> None:
    """Lazily build the EXTERN_OPS set from torch.ops.aten.

    Called once on first use.  Keeps the module importable even when
    torch.ops is not fully initialised at import time.
    """
    if EXTERN_OPS:
        return

    aten = torch.ops.aten

    # GEMM (CUBLAS)
    _gemm = [
        getattr(aten, "mm", None),
        getattr(aten, "bmm", None),
        getattr(aten, "addmm", None),
    ]

    # Attention (Flash / Efficient / SDPA)
    _attn = [
        getattr(aten, "_scaled_dot_product_flash_attention", None),
        getattr(aten, "_scaled_dot_product_efficient_attention", None),
        getattr(aten, "_flash_attention_forward", None),
        getattr(aten, "_efficient_attention_forward", None),
        getattr(aten, "_scaled_dot_product_flash_attention_backward", None),
        getattr(aten, "_scaled_dot_product_efficient_attention_backward", None),
        getattr(aten, "_flash_attention_backward", None),
        getattr(aten, "_efficient_attention_backward", None),
    ]

    # Convolution (cuDNN)
    _conv = [
        getattr(aten, "convolution", None),
        getattr(aten, "convolution_backward", None),
    ]

    # BatchNorm
    _bn = [
        getattr(aten, "_native_batch_norm_legit_functional", None),
        getattr(aten, "native_batch_norm_backward", None),
    ]

    for group in (_gemm, _attn, _conv, _bn):
        for op_or_packet in group:
            if op_or_packet is None:
                continue
            # op_or_packet may be an OpOverloadPacket (e.g. aten.mm)
            # We need the .default overload as that's what appears in FX graphs
            if hasattr(op_or_packet, "default"):
                EXTERN_OPS.add(op_or_packet.default)
            else:
                EXTERN_OPS.add(op_or_packet)


def is_extern_op(node: fx.Node) -> bool:
    """Return True if *node* corresponds to an Inductor ExternKernel op."""
    _populate_extern_ops()
    return getattr(node, "target", None) in EXTERN_OPS


def is_fusable_op(node: fx.Node) -> bool:
    """Return True if *node* is a compute op that Inductor would lower to a
    fusable Triton kernel (pointwise / reduction)."""
    if node.op not in ("call_function", "call_method"):
        return False
    _populate_extern_ops()
    return getattr(node, "target", None) not in EXTERN_OPS
