import torch
from torch.library import custom_op

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
def inject_layer_tags(model):
    """
    遍历模型的 encoder_layers 和 decoder_layers，自动打上 layer_rank 的标记。
    encoder 层编号从 0 开始，decoder 层编号紧接在 encoder 之后。
    """
    for i, layer in enumerate(model.encoder_layers):
        def pre_hook(module, args, rank=i):
            x = args[0]
            if isinstance(x, torch.Tensor):
                tagged_x = torch.ops.my_compiler.mark_layer(x, rank)
                return (tagged_x,) + args[1:]
            return args
        layer.register_forward_pre_hook(pre_hook)

    n_encoder = len(model.encoder_layers)
    for i, layer in enumerate(model.decoder_layers):
        def pre_hook(module, args, rank=n_encoder + i):
            x = args[0]
            if isinstance(x, torch.Tensor):
                tagged_x = torch.ops.my_compiler.mark_layer(x, rank)
                return (tagged_x,) + args[1:]
            return args
        layer.register_forward_pre_hook(pre_hook)

# 3. 在你的 main.py 中调用它 (在 graph_capture 之前)
# transformer = Transformer(...)
# inject_layer_tags(transformer)  <--- 加入这一行
# graph_capture = GraphCapture(transformer, src_data, tgt_data[:, :-1])