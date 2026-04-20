import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _make_checkpoint_forward(original_forward, use_reentrant: bool):
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=use_reentrant, **kwargs)

    return checkpointed_forward


def wrap_with_checkpoint(model: nn.Module, block_class_name: str, use_reentrant: bool = False) -> nn.Module:
    for module in model.modules():
        if module.__class__.__name__ != block_class_name:
            continue
        if hasattr(module, "_original_forward"):
            continue
        original_forward = module.forward
        module._original_forward = original_forward
        module.forward = _make_checkpoint_forward(original_forward, use_reentrant)
    return model


def unwrap_checkpoint(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
    return model
