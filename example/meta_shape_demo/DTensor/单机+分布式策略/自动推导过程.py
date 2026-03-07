import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
from torch._functorch.aot_autograd import aot_module_simplified
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode

# ================= 1. 窃听器：定义底层算子调度拦截模式 =================
class DTensorTraceMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        
        # 过滤掉一些无关紧要的内部算子，保持日志清晰
        if func.overloadpacket.__name__ in ["sym_size", "sym_stride", "view", "detach", "clone"]:
            return func(*args, **kwargs)

        print(f"\n🔍 [推导留痕] 执行底层算子: {func}")

        # 探测输入状态
        for i, arg in enumerate(args):
            if isinstance(arg, DTensor):
                print(f"   ├─ 输入 {i} 状态: DTensor (Fake), Placements={arg.placements}")

        # 让算子真正执行 (这里的执行是 Fake 级别的推导)
        out = func(*args, **kwargs)

        # 探测推导后的输出状态
        if isinstance(out, DTensor):
            print(f"   └─ 引擎推导输出: DTensor (Fake), Placements={out.placements}")
             
        return out

# ================= 2. 环境与模型初始化 =================
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
if not dist.is_initialized():
    dist.init_process_group("gloo")

mesh = DeviceMesh("cpu", [0])

class MyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(1024, 4096)
        self.out = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.out(torch.relu(self.wq(x)))

model = MyTransformerBlock()
tp_plan = {"wq": ColwiseParallel(), "out": RowwiseParallel()}
model = parallelize_module(model, mesh, tp_plan)

# ================= 3. 编译器拦截 =================

# 【修复点 1】：补充必填的前向和反向图接收器（空壳即可）
def simple_fw_compiler(gm, example_inputs):
    return gm.forward

def simple_bw_compiler(gm, example_inputs):
    return gm.forward

def my_dynamo_compiler(gm, example_inputs):
    print("\n>>> Dynamo 捕获完毕，AOTAutograd Fake 执行开始 <<<")
    print(">>> 窃听器已启动，正在记录推导留痕... <<<")
    
    with DTensorTraceMode():
        # 【修复点 2】：把 fw_compiler 和 bw_compiler 传给底层引擎
        compiled_aot = aot_module_simplified(
            gm, 
            example_inputs,
            fw_compiler=simple_fw_compiler,
            bw_compiler=simple_bw_compiler
        )
        
    print("\n>>> AOTAutograd Fake 执行结束，联合图已生成并切分 <<<")
    return compiled_aot

# ================= 4. 触发执行 =================
x_local = torch.randn(2, 1024, requires_grad=True)
x_dtensor = distribute_tensor(x_local, mesh, [Replicate()])

compiled_model = torch.compile(model, backend=my_dynamo_compiler)

print("\n--- 开始执行前向传播 ---")
out = compiled_model(x_dtensor)

print("\n--- 开始执行反向传播 ---")
out.sum().backward()

dist.destroy_process_group()