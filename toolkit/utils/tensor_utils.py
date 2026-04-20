import torch
import torch.nn as nn
from torch.fx.experimental.symbolic_shapes import hint_int


def _sym_scalar_bytes(val) -> int | None:
    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return 0
    return None


def val_bytes(val, fallback_numel: int = 4096) -> int:
    """Estimate bytes for Tensor-like FX meta values using hint_int for symbolic shapes."""
    if isinstance(val, torch.Tensor):
        numel = hint_int(val.numel(), fallback=fallback_numel)
        return int(numel) * val.element_size()

    if isinstance(val, (tuple, list)):
        return sum(val_bytes(v, fallback_numel=fallback_numel) for v in val)

    sym_bytes = _sym_scalar_bytes(val)
    if sym_bytes is not None:
        return sym_bytes

    return 0


def count_unique_params(model: nn.Module) -> int:
    """Count unique parameter bytes by traversing module._parameters directly."""
    seen_ptrs: set[int] = set()
    total_bytes = 0

    for module in model.modules():
        for param in module._parameters.values():
            if param is None:
                continue
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            total_bytes += param.numel() * param.element_size()

    return total_bytes
