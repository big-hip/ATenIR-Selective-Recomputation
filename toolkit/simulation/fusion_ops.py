"""Operator classification for fusion-aware memory estimation (L2.5).

Inductor fuses many pointwise ops into Triton kernels, eliminating intermediate
tensor allocations.  This module classifies ATen IR ops into *extern*
(non-fusable, e.g. mm/bmm/sdpa), *fusable* (a conservative allowlist of simple
pointwise ops), and conservative barriers (unknown/materializing ops).

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
FUSABLE_OPS: set = set()


def _add_overload(target_set: set, packet, overload: str) -> None:
    if packet is None:
        return
    op = getattr(packet, overload, None)
    if op is not None:
        target_set.add(op)


def _add_packet_overloads(target_set: set, aten, name: str, overloads: tuple[str, ...]) -> None:
    packet = getattr(aten, name, None)
    for overload in overloads:
        _add_overload(target_set, packet, overload)


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


def _populate_fusable_ops() -> None:
    """Lazily build a conservative pointwise allowlist.

    Unknown ops deliberately default to non-fusable.  This prevents L2.5 from
    over-eliminating allocations for materializing ops such as embedding/gather,
    clone, random ops, scatter, and layout-changing kernels.
    """
    if FUSABLE_OPS:
        return

    aten = torch.ops.aten

    # Unary pointwise ops.
    for name in (
        "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "erf", "erfc",
        "exp", "expm1", "floor", "frac", "log", "log10", "log1p", "log2",
        "neg", "reciprocal", "relu", "round", "rsqrt", "sigmoid", "sign",
        "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    ):
        _add_packet_overloads(FUSABLE_OPS, aten, name, ("default",))

    # Activation-like pointwise ops commonly present after decomposition.
    for name in ("gelu", "silu", "hardsigmoid", "hardswish", "leaky_relu"):
        _add_packet_overloads(FUSABLE_OPS, aten, name, ("default",))

    # Binary / scalar pointwise ops.
    for name in ("add", "sub", "mul", "div", "pow", "maximum", "minimum"):
        _add_packet_overloads(
            FUSABLE_OPS, aten, name,
            ("Tensor", "Scalar", "Tensor_Scalar", "Scalar_Tensor"),
        )

    # Comparisons and masks are pointwise and safe to keep in fusion groups.
    for name in ("eq", "ne", "lt", "le", "gt", "ge"):
        _add_packet_overloads(FUSABLE_OPS, aten, name, ("Tensor", "Scalar"))
    _add_packet_overloads(FUSABLE_OPS, aten, "where", ("self",))

    # Backward pointwise kernels after decomposition.
    for name in (
        "threshold_backward", "sigmoid_backward", "tanh_backward",
        "gelu_backward", "silu_backward",
    ):
        _add_packet_overloads(FUSABLE_OPS, aten, name, ("default",))


def is_extern_op(node: fx.Node) -> bool:
    """Return True if *node* corresponds to an Inductor ExternKernel op."""
    _populate_extern_ops()
    return getattr(node, "target", None) in EXTERN_OPS


def is_fusable_op(node: fx.Node) -> bool:
    """Return True if *node* is in the conservative pointwise allowlist."""
    if node.op not in ("call_function", "call_method"):
        return False
    _populate_extern_ops()
    _populate_fusable_ops()
    target = getattr(node, "target", None)
    return target in FUSABLE_OPS and target not in EXTERN_OPS


def is_fusion_barrier(node: fx.Node) -> bool:
    """Return True for materializing ops that should break L2.5 fusion groups."""
    if node.op not in ("call_function", "call_method"):
        return True
    return is_extern_op(node) or not is_fusable_op(node)
