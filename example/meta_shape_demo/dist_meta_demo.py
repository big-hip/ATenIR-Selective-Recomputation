"""
dist_meta_demo.py
=================
验证：在 FX 图上做分布式变换后，meta['val'] 能否通过 FakeTensorProp 正确刷新。

Step 1  追踪原始图，查看初始 meta['val']
Step 2  直接改某节点的 meta['val'] ── 证明不会传播到下游
Step 3  对已有图插入 slice（模拟列切分）→ FakeTensorProp 刷新 ── 下游跟随变化
Step 4  在已有图中插入自定义通信算子 all_gather ── FakeTensorProp 能正确推导扩展后形状

使用的模型（单层，形状变化最清晰）：
    x (4, 64) → addmm (4, 128) → relu (4, 128) → output
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


# ─────────────────────────────────────────────────────────────────────────────
# 演示用模型（单层，形状变化清晰）
# ─────────────────────────────────────────────────────────────────────────────

class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 128)

    def forward(self, x):           # x: (4, 64)
        h = self.fc(x)              # (4, 128)
        return torch.relu(h)        # (4, 128)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def print_shapes(gm: fx.GraphModule, title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"{'node name':<35} {'op':<18} shape")
    print("-" * 62)
    for node in gm.graph.nodes:
        val = node.meta.get('val', None)
        if isinstance(val, torch.Tensor):
            shape = str(tuple(val.shape))
        elif val is not None:
            shape = repr(val)[:40]
        else:
            shape = "—"
        print(f"{node.name:<35} {node.op:<18} {shape}")


def find_node(gm: fx.GraphModule, name_substr: str) -> fx.Node:
    for n in gm.graph.nodes:
        if name_substr in n.name:
            return n
    raise KeyError(f"找不到名称含 '{name_substr}' 的节点")


def reprop(gm: fx.GraphModule, sample_input: torch.Tensor) -> None:
    """用 FakeTensorProp 重新传播整张图的 meta['val']。"""
    with FakeTensorMode(allow_non_fake_inputs=True) as fm:
        fake_x = fm.from_tensor(sample_input)
        FakeTensorProp(gm, mode=fm).propagate(fake_x)


def fresh_graph() -> fx.GraphModule:
    """追踪一张干净的 OneLayerModel 图，带完整 meta['val']。"""
    model = OneLayerModel().eval()
    with FakeTensorMode(allow_non_fake_inputs=True) as fm:
        gm = make_fx(model, tracing_mode="fake")(fm.from_tensor(torch.zeros(4, 64)))
    return gm


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：追踪原始图
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n【Step 1】追踪原始图，查看 meta['val']")
gm_orig = fresh_graph()
print_shapes(gm_orig, "原始图  x(4,64) → addmm(4,128) → relu(4,128)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：直接修改 addmm 的 meta['val']，不重传播
#          结论：下游 relu 的 meta['val'] 不会跟着变
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n【Step 2】直接改 addmm 的 meta['val'] → 下游不自动更新")

gm2 = fresh_graph()
addmm_node = find_node(gm2, "addmm")
relu_node  = find_node(gm2, "relu")

print(f"\n  修改前: addmm={tuple(addmm_node.meta['val'].shape)}"
      f"  relu={tuple(relu_node.meta['val'].shape)}")

with FakeTensorMode(allow_non_fake_inputs=True) as fm:
    addmm_node.meta['val'] = fm.from_tensor(torch.zeros(4, 32))   # 假装切分了

print(f"  修改后: addmm={tuple(addmm_node.meta['val'].shape)}"
      f"  relu={tuple(relu_node.meta['val'].shape)}  ← 没变！")

print_shapes(gm2, "直接修改 meta['val'] 后（relu 仍是旧形状）")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3：在已有图中插入 slice 节点（模拟列切分 128→32）
#          然后用 FakeTensorProp 重传播 → 下游 relu 跟着变成 (4, 32)
#
#  变换前: addmm(4,128) → relu(4,128) → output
#  变换后: addmm(4,128) → slice[:,0:32](4,32) → relu(4,32) → output
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n【Step 3】插入 slice 节点（列切分模拟）→ FakeTensorProp 刷新下游形状")

gm3 = fresh_graph()
addmm3 = find_node(gm3, "addmm")
relu3  = find_node(gm3, "relu")

print(f"\n  插入前: addmm={tuple(addmm3.meta['val'].shape)}"
      f"  relu={tuple(relu3.meta['val'].shape)}")

# ── 在 addmm 之后插入 slice(dim=1, start=0, end=32) ──
with gm3.graph.inserting_after(addmm3):
    slice_node = gm3.graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(addmm3, 1, 0, 32),   # tensor, dim, start, end
    )

# 让 relu 消费 slice 的输出而不是 addmm 的输出
relu3.replace_input_with(addmm3, slice_node)

gm3.graph.lint()
gm3.recompile()

print("  图结构已修改：addmm → slice[:,0:32] → relu → output")
print("  ── 重传播前 ──")
print(f"  slice meta['val'] = {slice_node.meta.get('val', '（未填充）')}")
print(f"  relu  meta['val'] = {tuple(relu3.meta['val'].shape)}  ← 仍是旧形状")

# ── 重传播 ──
reprop(gm3, torch.zeros(4, 64))

print("\n  ── FakeTensorProp 重传播后 ──")
print_shapes(gm3, "插入 slice + 重传播（relu 已跟随更新为 (4,32)）")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4：插入自定义通信算子 all_gather，FakeTensorProp 能推导扩展后形状
#
#  通信算子特点：输入 (4, 32) → all_gather(world_size=4) → 输出 (16, 32)
#               即 dim0 × world_size，形状扩张
#
#  关键：必须给算子注册 register_fake（abstract impl），
#        FakeTensorProp 才能在不实际通信的情况下推导出输出形状。
#
#  变换后图:
#    addmm(4,128) → slice(4,32) → all_gather(16,32) → relu(16,32) → output
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n【Step 4】插入自定义 all_gather 通信算子 → FakeTensorProp 推导扩展形状")

# ── 定义通信算子（含 FakeTensor abstract 实现）──────────────────────────────
@torch.library.custom_op("demo_dist::all_gather_tensor", mutates_args=())
def all_gather_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    模拟 all_gather：从 world_size 个 rank 收集张量，沿 dim=0 拼接。
    真实场景中这里会是 torch.distributed.all_gather。
    """
    return x.repeat(world_size, 1)      # 仅用于 CPU eager 执行，非真实通信

@all_gather_tensor.register_fake
def _all_gather_tensor_fake(x, world_size):
    """
    FakeTensor 下的形状推导：dim0 × world_size，其余维度不变。
    FakeTensorProp 会调用这个函数而不是真正执行 all_gather。
    """
    new_shape = (x.shape[0] * world_size,) + tuple(x.shape[1:])
    return x.new_empty(new_shape)

# ── 在 Step 3 的图基础上继续，接着 slice 节点插入 all_gather ──────────────
# 直接复用 gm3（已经有 slice 节点），在 slice 后面插入 all_gather

WORLD_SIZE = 4

with gm3.graph.inserting_after(slice_node):
    comm_node = gm3.graph.call_function(
        torch.ops.demo_dist.all_gather_tensor,
        args=(slice_node, WORLD_SIZE),
    )

# relu 改为消费 comm_node 的输出
relu3.replace_input_with(slice_node, comm_node)

gm3.graph.lint()
gm3.recompile()

print(f"\n  图结构：addmm → slice[:,0:32] → all_gather(world_size={WORLD_SIZE}) → relu → output")
print("  ── 重传播前 ──")
print(f"  comm_node meta['val'] = {comm_node.meta.get('val', '（未填充）')}")
print(f"  relu      meta['val'] = {tuple(relu3.meta['val'].shape)}  ← 仍是旧形状")

# ── 重传播 ──
reprop(gm3, torch.zeros(4, 64))

print("\n  ── FakeTensorProp 重传播后 ──")
print_shapes(gm3, f"插入 all_gather(world_size={WORLD_SIZE}) + 重传播")

comm_shape = tuple(comm_node.meta['val'].shape)
relu_shape  = tuple(relu3.meta['val'].shape)
print(f"\n  all_gather 输出: {comm_shape}  ← (4×{WORLD_SIZE}=16, 32) ✓")
print(f"  relu       输出: {relu_shape}   ← 跟随 all_gather 正确更新 ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 总结
# ─────────────────────────────────────────────────────────────────────────────

print("""
┌────────────────────────────────────────────────────────────┐
│  结论                                                      │
│                                                            │
│  Step 2  直接改 meta['val']              ✗ 不传播          │
│          下游节点仍持有旧形状                              │
│                                                            │
│  Step 3  插入新节点 + FakeTensorProp     ✓ 下游全部刷新    │
│          .propagate(*fake_inputs)                          │
│          插入 slice 后 relu 从(4,128)→(4,32)               │
│                                                            │
│  Step 4  插入通信算子 + register_fake    ✓ 形状扩张正确    │
│          all_gather: (4,32)→(16,32)                        │
│          relu 跟随更新为 (16,32)                           │
│                                                            │
│  实践要点：                                                │
│    1. 所有新插入的算子必须有 register_fake（abstract impl） │
│       FakeTensorProp 才能推导其输出形状                    │
│    2. 插入/修改后调用 FakeTensorProp.propagate             │
│       整张图的 meta['val'] 一次性全部刷新                  │
└────────────────────────────────────────────────────────────┘
""")
