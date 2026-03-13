import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard, Replicate
from torch.fx.experimental.proxy_tensor import make_fx

# 1. 简易本地伪分布式环境初始化 (纯 CPU 单进程模拟)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
dist.init_process_group("gloo")

# 2. 构建 DeviceMesh (1维网格，代表我们要在这组设备上做张量并行)
mesh = DeviceMesh("cpu", [0])

# 3. 定义我们想要被 AOTAutograd 追踪的“纯逻辑”函数
def tp_matmul_simulate(x_logical, w_logical):
    # 第一步：逻辑上的矩阵相乘
    out = torch.matmul(x_logical, w_logical)
    
    # 第二步：TP（张量并行）规范。
    # 因为 x 按列切，w 按行切，乘出来的结果在各卡上只是“部分和 (Partial Sum)”。
    # 我们需要将其 redistribute 回 Replicate (全量复制) 状态，这会强行触发 All-Reduce 通信。
    return out.redistribute(placements=[Replicate()])

# 4. 创建普通的本地 Tensor
x_local = torch.randn(4, 8)
w_local = torch.randn(8, 4)

# 5. DTensor 包装魔法：赋予分布式布局 (Placement)
# 模拟典型张量并行 (TP)：激活值 x 在维度 1 (列) 切分，权重 w 在维度 0 (行) 切分
x_dtensor = distribute_tensor(x_local, mesh, [Shard(1)])
w_dtensor = distribute_tensor(w_local, mesh, [Shard(0)])

# 6. 利用 make_fx (AOTAutograd的核心追踪器) 进行算子拦截与图展开
print("开始追踪...")
traced_graph = make_fx(tp_matmul_simulate)(x_dtensor, w_dtensor)

# 7. 打印底层的物理计算图代码
print("\n[底层真实的物理计算与通信图 (ATen IR)]:")
print(traced_graph.code)

# 清理环境
dist.destroy_process_group()