"""
meta['val'] 形状自动推导演示
=============================
PyTorch FX 图中每个节点的 meta['val'] 是一个 FakeTensor，
它携带了该节点输出的 dtype / shape / device 信息，
而不需要真正执行计算。

本 demo 展示三种获取方式：
  1. make_fx + FakeTensorMode  （最底层，手动触发）
  2. aot_module_simplified      （AOT Autograd，meta 由框架自动填充）
  3. ShapeProp                  （给已有 FX 图补充 shape 信息）
"""

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from functorch.compile import aot_module_simplified, make_boxed_func
from torch.fx.passes.shape_prop import ShapeProp


# ─────────────────────────────────────────────────────────────────────────────
# 演示用简单模型
# ─────────────────────────────────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 32)
        self.ln      = nn.LayerNorm(128)

    def forward(self, x):           # x: (B, 64)
        h = self.linear1(x)         # (B, 128)
        h = self.ln(h)              # (B, 128)
        h = torch.relu(h)           # (B, 128)
        h = self.linear2(h)         # (B, 32)
        return h


def print_graph_shapes(gm: fx.GraphModule, title: str):
    """打印图中每个节点的 name / op / shape。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'node name':<30} {'op':<15} {'shape / value'}")
    print("-" * 60)
    for node in gm.graph.nodes:
        val = node.meta.get('val', None)
        if isinstance(val, torch.Tensor):
            shape_str = str(tuple(val.shape))
        elif val is not None:
            shape_str = repr(val)
        else:
            shape_str = "—"
        print(f"{node.name:<30} {node.op:<15} {shape_str}")


# ─────────────────────────────────────────────────────────────────────────────
# 方法 1：make_fx + FakeTensorMode
# ─────────────────────────────────────────────────────────────────────────────

def demo_make_fx():
    """
    make_fx 在 FakeTensorMode 下追踪模型，得到的图每个节点的
    meta['val'] 自动是 FakeTensor，包含正确 shape 但不占实际显存。
    """
    print("\n\n【方法 1】make_fx + FakeTensorMode")

    model = TinyModel().eval()

    # 构造 FakeTensor 输入（不分配真实内存）
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_x = fake_mode.from_tensor(torch.zeros(4, 64))
        # make_fx 会在 FakeTensorMode 下运行 forward，自动推导每个中间节点的 shape
        gm = make_fx(model, tracing_mode="fake")(fake_x)

    print_graph_shapes(gm, "make_fx 捕获的图（FakeTensorMode）")

    # 演示：从 meta['val'] 读出某节点的 shape
    for node in gm.graph.nodes:
        if node.op == 'call_function' and 'linear' in node.name:
            val = node.meta.get('val')
            if val is not None:
                print(f"\n  节点 {node.name!r} 的输出 shape = {tuple(val.shape)}"
                      f"，dtype = {val.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# 方法 2：AOT Autograd（aot_module_simplified）
# ─────────────────────────────────────────────────────────────────────────────

def demo_aot():
    """
    AOT Autograd 会同时生成 FW 图和 BW 图，
    两张图的节点都带有 meta['val']（FakeTensor）。
    这是项目本身的工作方式。
    """
    print("\n\n【方法 2】AOT Autograd（aot_module_simplified）")

    model  = TinyModel().train()
    x      = torch.randn(4, 64)

    # PyTorch 2.6 要求 aot_module_simplified 的模型输出必须是 tuple
    class _TupleWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return (self.m(x),)

    wrapped = _TupleWrapper(model)
    captured = {}

    def fw_compiler(gm, sample_inputs):
        captured['fw'] = gm
        return make_boxed_func(gm.forward)

    def bw_compiler(gm, sample_inputs):
        captured['bw'] = gm
        return make_boxed_func(gm.forward)

    # aot_module_simplified 追踪并分区，fw/bw compiler 回调时图已有 meta['val']
    compiled = aot_module_simplified(wrapped, [x], fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    # 触发 FW 捕获（wrapped 输出是 list，取 [0]）
    out = compiled(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    # 触发 BW 捕获
    out.sum().backward()

    if 'fw' in captured:
        print_graph_shapes(captured['fw'], "AOT FW 图")
    if 'bw' in captured:
        print_graph_shapes(captured['bw'], "AOT BW 图")


# ─────────────────────────────────────────────────────────────────────────────
# 方法 3：ShapeProp（给已有图补充 shape 信息）
# ─────────────────────────────────────────────────────────────────────────────

def demo_shape_prop():
    """
    如果你已经有一张 symbolic_trace 得到的普通 FX 图（没有 meta['val']），
    可以用 ShapeProp 跑一遍真实前向来填充 shape。
    注意：ShapeProp 写入的是 meta['tensor_meta']，不是 meta['val']。
    """
    print("\n\n【方法 3】ShapeProp（补充 meta['tensor_meta']）")

    model = TinyModel().eval()
    x     = torch.randn(4, 64)

    # symbolic_trace 不含 shape 信息
    gm = fx.symbolic_trace(model)

    # ShapeProp 用真实输入跑一遍，把 shape 写入每个节点的 meta
    ShapeProp(gm).propagate(x)

    print(f"\n{'node name':<30} {'op':<15} {'tensor_meta.shape'}")
    print("-" * 60)
    for node in gm.graph.nodes:
        tm = node.meta.get('tensor_meta', None)
        if tm is not None:
            print(f"{node.name:<30} {node.op:<15} {tuple(tm.shape)}")
        else:
            print(f"{node.name:<30} {node.op:<15} —")


# ─────────────────────────────────────────────────────────────────────────────
# 实用工具：估算节点激活内存（用 meta['val']）
# ─────────────────────────────────────────────────────────────────────────────

def estimate_activation_memory(gm: fx.GraphModule) -> None:
    """
    遍历图中所有 call_function / call_method 节点，
    从 meta['val'] 读取 shape 和 dtype，估算激活占用字节数。
    """
    print(f"\n{'─'*60}")
    print("  激活内存估算（基于 meta['val']）")
    print(f"{'─'*60}")

    dtype_bytes = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.float64: 8, torch.int32: 4, torch.int64: 8,
    }

    total = 0
    for node in gm.graph.nodes:
        if node.op not in ('call_function', 'call_method'):
            continue
        val = node.meta.get('val')
        if not isinstance(val, torch.Tensor):
            continue
        nbytes = val.numel() * dtype_bytes.get(val.dtype, 4)
        total += nbytes
        print(f"  {node.name:<28} shape={str(tuple(val.shape)):<20} "
              f"{nbytes / 1024:.1f} KB")

    print(f"\n  合计激活内存（静态估算）: {total / 1024 / 1024:.3f} MB")


if __name__ == "__main__":
    demo_make_fx()

    # make_fx 的图可以直接拿来做内存估算
    model = TinyModel().eval()
    with FakeTensorMode(allow_non_fake_inputs=True):
        gm = make_fx(model, tracing_mode="fake")(torch.zeros(4, 64))
    estimate_activation_memory(gm)

    demo_aot()
    demo_shape_prop()
