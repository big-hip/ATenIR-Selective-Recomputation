from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)


aten = torch.ops.aten

SAC_POLICIES = {
    "save_matmuls": lambda ctx, op, *args, **kwargs: (
        CheckpointPolicy.MUST_SAVE
        if op in {aten.mm.default, aten.addmm.default, aten.bmm.default}
        else CheckpointPolicy.PREFER_RECOMPUTE
    ),
    "save_attention": lambda ctx, op, *args, **kwargs: (
        CheckpointPolicy.MUST_SAVE
        if op in {
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten._scaled_dot_product_flash_attention.default,
            aten._scaled_dot_product_efficient_attention.default,
        }
        else CheckpointPolicy.PREFER_RECOMPUTE
    ),
    "recompute_all": lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE,
}


def _make_sac_forward(original_forward, policy_fn):
    context_fn = partial(create_selective_checkpoint_contexts, policy_fn)

    def checkpointed_forward(*args, **kwargs):
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            context_fn=context_fn,
            **kwargs,
        )

    return checkpointed_forward


def wrap_with_sac(model: nn.Module, block_class_name: str, policy_name: str = "save_matmuls") -> nn.Module:
    try:
        policy_fn = SAC_POLICIES[policy_name]
    except KeyError as exc:
        raise ValueError(f"Unknown SAC policy: {policy_name}. Choose from {list(SAC_POLICIES)}") from exc

    for module in model.modules():
        if module.__class__.__name__ != block_class_name:
            continue
        if hasattr(module, "_original_forward"):
            continue
        original_forward = module.forward
        module._original_forward = original_forward
        module.forward = _make_sac_forward(original_forward, policy_fn)
    return model
