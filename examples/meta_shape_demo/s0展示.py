import torch

def simple_model(x):
    return x * 2

dummy_input = torch.randn(4, 10)

# ---------------------------------------------------------
# 实验 A: 默认模式 (先特化)
# ---------------------------------------------------------
# 注意查看输出，维度 0 会被写死为 4
print("=== 实验 A: 默认行为 (静态特化) ===")
opt_model_A = torch.compile(simple_model)
# 触发一次底层编译以查看图 (由于无法直接打印 compiled graph, 我们用解释机制)
torch._dynamo.explain(opt_model_A)(dummy_input) 

# ---------------------------------------------------------
# 实验 B: 强制开启 dynamic=True (第一次即泛化)
# ---------------------------------------------------------
# 注意查看输出，维度 0 会变成 s0
print("\n=== 实验 B: dynamic=True (首次捕获即为 s0) ===")
opt_model_B = torch.compile(simple_model, dynamic=True)
torch._dynamo.explain(opt_model_B)(dummy_input)

# ---------------------------------------------------------
# 实验 C: 使用底层 API mark_dynamic 精准打击
# ---------------------------------------------------------
print("\n=== 实验 C: mark_dynamic (精准控制单个维度) ===")
# 我们显式告诉底层：第 0 维是动态的
torch._dynamo.mark_dynamic(dummy_input, 0)
opt_model_C = torch.compile(simple_model) # 即使不加 dynamic=True
torch._dynamo.explain(opt_model_C)(dummy_input)