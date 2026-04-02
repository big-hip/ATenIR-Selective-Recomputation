#输出重定向TORCH_LOGS="dynamo,aot,graph,output_code" python test_compile.py 2>&1 | tee output.log
"""
For more info on various settings, try TORCH_LOGS="help"
Valid settings:
all, cache, dynamo, fake_tensor, aot, autograd, inductor, dynamic, torch, distributed, c10d, ddp, pp, fsdp, dtensor, onnx, export, sym_node, custom_format_test_artifact, compiled_autograd_verbose, recompiles_verbose, onnx_diagnostics, graph_sizes, post_grad_graphs, kernel_code, aot_graphs_effects, schedule, loop_ordering, not_implemented, ddp_graphs, aot_graphs, guards, graph, perf_hints, verbose_guards, trace_bytecode, overlap, fusion, trace_shape_events, aot_joint_graph, cudagraph_static_inputs, graph_region_expansion, graph_breaks, trace_call, bytecode, compiled_autograd, trace_source, recompiles, benchmarking, graph_code, output_code, cudagraphs

TORCH_LOGS="all" python test_compile.py 2>&1 | tee output.log
"""

import torch
import torch.nn as nn

# 1. 定义包含 Linear (带 Bias) + ReLU 的模型
class LinearReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 开启 bias
        self.linear = nn.Linear(10, 10, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算流：Input -> (Linear: xW + b) -> ReLU
        x = self.linear(x)
        return self.relu(x)

# 2. 初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
# 建议开启 TF32 以消除警告（针对 A100）
torch.set_float32_matmul_precision('high')

model = LinearReLUModel().to(device)
x = torch.randn(16, 10, device=device, requires_grad=True)

# 3. 编译
compiled_model = torch.compile(model)

# 4. 执行
print("=== 开始执行（带 Bias 模式） ===")
out = compiled_model(x)
loss = out.sum()
loss.backward()
print("=== 编译与执行结束 ===")