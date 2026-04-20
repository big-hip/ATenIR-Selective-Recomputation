"""Memory budget helpers for Inductor's activation_memory_budget.

Note: uses module-level globals — NOT thread-safe.  Only call from a
single thread (which matches the typical torch.compile workflow).
"""

_PREVIOUS_BUDGET = None
_HAS_ACTIVE_OVERRIDE = False


def set_memory_budget(budget: float = 0.5) -> bool:
    global _PREVIOUS_BUDGET, _HAS_ACTIVE_OVERRIDE

    import torch._functorch.config as cfg

    try:
        if not _HAS_ACTIVE_OVERRIDE:
            _PREVIOUS_BUDGET = cfg.activation_memory_budget
        cfg.activation_memory_budget = budget
        _HAS_ACTIVE_OVERRIDE = True
        return True
    except Exception:
        return False


def clear_memory_budget():
    global _PREVIOUS_BUDGET, _HAS_ACTIVE_OVERRIDE

    if not _HAS_ACTIVE_OVERRIDE:
        return

    import torch._functorch.config as cfg

    cfg.activation_memory_budget = _PREVIOUS_BUDGET
    _PREVIOUS_BUDGET = None
    _HAS_ACTIVE_OVERRIDE = False
