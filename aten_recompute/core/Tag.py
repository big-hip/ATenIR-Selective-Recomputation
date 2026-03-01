import torch
from torch.library import custom_op
from torch.utils.hooks import RemovableHandle
from typing import Iterable, List, Tuple
import torch.nn as nn

# 1. 注册一个自定义的 ATen 算子：接受 Tensor 和层号，返回 Tensor
@custom_op("my_compiler::mark_layer", mutates_args=())
def mark_layer(x: torch.Tensor, layer_rank: int) -> torch.Tensor:
    # 使用 clone 确保算子不会被 AOT 优化器折叠掉
    return x.clone()


# 注册 Fake Tensor 推导规则（AOT Autograd 编译时需要）
@mark_layer.register_fake
def _mark_layer_fake(x: torch.Tensor, layer_rank: int) -> torch.Tensor:
    return torch.empty_like(x)


# 注册 Autograd 规则：该算子在数学上等价于 identity，对 x 的梯度直接透传
def _mark_layer_setup_context(ctx, inputs, output):
    # 这里不需要保存任何中间量，留空即可
    return


def _mark_layer_backward(ctx, grad_output):
    # 对第一个输入 x 的梯度为 grad_output，本身是恒等映射；
    # 对第二个输入 layer_rank（int 标量）不需要梯度，返回 None。
    return grad_output, None


torch.library.register_autograd(
    "my_compiler::mark_layer",
    _mark_layer_backward,
    setup_context=_mark_layer_setup_context,
)

# 2. 编写 Hook 注入函数
def inject_layer_tags(
    layers: Iterable[Tuple[nn.Module, int]],
) -> List[RemovableHandle]:
    """
    为指定的层列表注入 layer_rank 前向钩子。

    Args:
        layers: 可迭代对象，每项为 (nn.Module, rank: int)。
                调用方自行构造层与 rank 的对应关系，与模型架构无关。

    Returns:
        List[RemovableHandle]：每个注册的钩子对应一个 handle。
        在测试或推理时调用 handle.remove() 可撤销所有钩子。

    Example（Encoder-Decoder Transformer）::

        enc = [(layer, i) for i, layer in enumerate(model.encoder_layers)]
        dec = [(layer, len(model.encoder_layers) + i)
               for i, layer in enumerate(model.decoder_layers)]
        handles = inject_layer_tags(enc + dec)

    Example（GPT-2 风格，只有 decoder）::

        handles = inject_layer_tags([(blk, i) for i, blk in enumerate(model.transformer.h)])
    """
    handles: List[RemovableHandle] = []
    for layer, rank in layers:
        def pre_hook(module, args, r=rank):
            x = args[0]
            if isinstance(x, torch.Tensor):
                tagged_x = torch.ops.my_compiler.mark_layer(x, r)
                return (tagged_x,) + args[1:]
            return args
        handles.append(layer.register_forward_pre_hook(pre_hook))
    return handles