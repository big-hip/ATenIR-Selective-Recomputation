import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate

# PyTorch 2.6 底层 AOT 核心 API
from torch._functorch.aot_autograd import aot_module_simplified
# 引入默认的图切分器，用于我们在拦截并打印联合图后，让其继续正常切分
from torch._functorch.partitioners import default_partition

# ================= 1. 分布式环境与模型初始化 =================
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
tp_plan = {
    "wq": ColwiseParallel(), 
    "out": RowwiseParallel(), 
}
model = parallelize_module(model, mesh, tp_plan)

# ================= 2. 核心：编译器多级拦截探针 =================

# 【拦截点 3】：切分后的前向图
def aot_fw_compiler(gm, example_inputs):
    print("\n" + "="*15 + " [阶段 3] AOTAutograd 切分后的前向图 (FW Graph) " + "="*15)
    print(gm.code)
    return gm.forward

# 【拦截点 4】：切分后的反向图
def aot_bw_compiler(gm, example_inputs):
    print("\n" + "="*15 + " [阶段 4] AOTAutograd 切分后的反向图 (BW Graph) " + "="*15)
    print(gm.code)
    return gm.forward

# 【核心拦截点 2】：拦截 Joint Graph (体现 FakeTensor 与 DTensor 分发对图的改变)
def my_partition_fn(joint_module, joint_inputs, **kwargs):
    print("\n" + "="*15 + " [阶段 2] 核心：联合图 Joint Graph (DTensor分发注入后) " + "="*15)
    print("💡 发生了什么：AOTAutograd 刚用 FakeTensor 跑完图。")
    print("💡 DTensor 机制已自动拦截了高阶矩阵乘法，推导出了真实的物理操作，并注入了底层的通信算子！")
    print(joint_module.code)
    
    # 打印完毕后，将联合图交还给官方切分器，让它切成前向和反向两张图
    return default_partition(joint_module, joint_inputs, **kwargs)

# 【拦截点 1】：前端初始逻辑图
def my_dynamo_compiler(gm, example_inputs):
    print("\n" + "="*15 + " [阶段 1] Dynamo 捕获的初始前向图 (纯逻辑) " + "="*15)
    print(gm.code)
    
    # 启动 AOTAutograd，并把我们的探针全部挂载进去
    return aot_module_simplified(
        gm, 
        example_inputs, 
        fw_compiler=aot_fw_compiler, 
        bw_compiler=aot_bw_compiler,
        partition_fn=my_partition_fn  # 必须通过挂载 partition_fn 才能拿到活的联合图
    )

# ================= 3. 触发编译 =================
x_local = torch.randn(2, 1024, requires_grad=True)
x_dtensor = distribute_tensor(x_local, mesh, [Replicate()])

compiled_model = torch.compile(model, backend=my_dynamo_compiler)

print("\n>>> 开始执行前向传播 (触发 Dynamo 与 AOT 编译) <<<")
out = compiled_model(x_dtensor)

print("\n>>> 开始执行反向传播 <<<")
out.sum().backward()

dist.destroy_process_group()