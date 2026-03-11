"""
checkpoint_wrapper.py

封装 PyTorch 原生 torch.utils.checkpoint，提供模型级别的 activation checkpoint 包装。
用于与 ATenIR 选择性重计算进行公平的内存/性能对比。

用法::

    from aten_recompute.utils.checkpoint_wrapper import apply_activation_checkpoint

    apply_activation_checkpoint(
        model,
        module_lists=[model.encoder_layers, model.decoder_layers],
    )
"""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Sequence, Tuple, Type

from .logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "wrap_layer_with_checkpoint",
    "apply_activation_checkpoint",
    "remove_activation_checkpoint",
]


def wrap_layer_with_checkpoint(
    module: nn.Module,
    use_reentrant: bool = False,
) -> None:
    """
    将单个 module 的 forward 包装为 checkpoint 调用。

    原始 forward 保存为 module._original_forward，便于后续恢复。
    """
    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )

    module._original_forward = original_forward
    module.forward = checkpointed_forward


def apply_activation_checkpoint(
    model: nn.Module,
    layer_types: Optional[Tuple[Type[nn.Module], ...]] = None,
    module_lists: Optional[Sequence[nn.ModuleList]] = None,
    use_reentrant: bool = False,
) -> int:
    """
    对模型中的指定层启用 PyTorch 原生 activation checkpoint。

    Parameters
    ----------
    model : 目标模型（未使用但保留参数以便 layer_types 方式遍历）
    layer_types : 要启用 checkpoint 的层类型元组
    module_lists : 要启用 checkpoint 的 ModuleList 列表（优先级高于 layer_types）
    use_reentrant : 是否使用 reentrant 模式（推荐 False）

    Returns
    -------
    启用 checkpoint 的层数量
    """
    count = 0

    if module_lists is not None:
        for ml in module_lists:
            for module in ml:
                wrap_layer_with_checkpoint(module, use_reentrant=use_reentrant)
                count += 1
    elif layer_types is not None:
        for _name, module in model.named_modules():
            if isinstance(module, layer_types):
                wrap_layer_with_checkpoint(module, use_reentrant=use_reentrant)
                count += 1
    else:
        logger.warning("[checkpoint_wrapper] 未指定 layer_types 或 module_lists，不启用 checkpoint。")

    logger.info("[checkpoint_wrapper] 已对 %d 个层启用 activation checkpoint。", count)
    return count


def remove_activation_checkpoint(model: nn.Module) -> int:
    """
    移除由 apply_activation_checkpoint 添加的 checkpoint 包装，恢复原始 forward。
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            count += 1
    logger.info("[checkpoint_wrapper] 已从 %d 个层移除 checkpoint 包装。", count)
    return count
