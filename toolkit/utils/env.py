"""Experiment environment setup: suppress third-party warnings, enable TF32.

Usage at top of every experiment script::

    from toolkit.utils import setup_experiment_env
    setup_experiment_env()
"""

import warnings


def setup_experiment_env(*, enable_tf32: bool = True) -> None:
    """Configure runtime for reproducible, clean experiment runs.

    - Filters known third-party warnings (transformers, dynamo PGO, inductor
      TF32 hint, SAC context_fn, tempfile ResourceWarning).
    - Optionally enables TF32 for faster matmul on Ampere+ GPUs.
      TF32 does NOT affect memory allocation sizes, so simulation MRE
      is unchanged.

    Args:
        enable_tf32: If True (default), call
            ``torch.set_float32_matmul_precision('high')`` to enable TF32.
    """
    # ── W1: transformers FutureWarning on _register_pytree_node ──
    warnings.filterwarnings(
        "ignore",
        message=r".*_register_pytree_node.*",
        category=FutureWarning,
    )

    # ── W2: TensorFloat32 hint from inductor ──
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat32 tensor cores.*",
        category=UserWarning,
    )

    # ── W3: dynamo_pgo force disabled ──
    warnings.filterwarnings(
        "ignore",
        message=r".*dynamo_pgo force disabled.*",
        category=UserWarning,
    )

    # ── W4: context_fn passed to checkpoint under torch.compile ──
    warnings.filterwarnings(
        "ignore",
        message=r".*context_fn is passed to.*checkpoint.*",
        category=UserWarning,
    )

    # ── W5: TemporaryDirectory cleanup (PyTorch inductor internal) ──
    warnings.filterwarnings(
        "ignore",
        message=r".*Implicitly cleaning up.*TemporaryDirectory.*",
        category=ResourceWarning,
    )

    # ── Enable TF32 ──
    if enable_tf32:
        import torch
        torch.set_float32_matmul_precision("high")
