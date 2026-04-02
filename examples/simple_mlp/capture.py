"""
capture.py — 捕获 SimpleMLP 的 ATen IR（联合图 + FW/BW 分离图）。

用法::

    cd examples/simple_mlp
    python capture.py                          # 默认策略 6
    RECOMPUTE='{"0": null}' python capture.py  # 不重计算
    RECOMPUTE='{"1": null}' python capture.py  # 全部重计算
    RECOMPUTE='{"sac": null}' python capture.py  # PyTorch SAC (内置 min-cut)
"""
import copy
import json
import functools
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")
os.environ.setdefault("SAVE_JOINT_GRAPH", "1")

import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.virtualized import V
from torch.utils.checkpoint import (
    checkpoint as torch_checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from model import SimpleMLP, device
from aten_recompute.core import CompilerBackend, inject_layer_tags
from aten_recompute.utils.save_ir import save_ir_and_dot

_LINE = "─" * 60
_BOLD = "═" * 60

MODEL_NAME = "SimpleMLP"
os.environ["MODEL_NAME"] = MODEL_NAME


# ── PyTorch SAC 相关 ──────────────────────────────────────────────────────────

_COMPUTE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
}


def _sac_policy(ctx, op, *args, **kwargs):
    """SAC 策略：保存计算密集算子，重计算廉价算子。"""
    if op in _COMPUTE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _wrap_model_with_sac(model):
    """为模型的每个 Linear 层包装 SAC checkpoint。"""
    sac_context_fn = functools.partial(
        create_selective_checkpoint_contexts, _sac_policy
    )

    # 收集所有 Linear 子模块并包装
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            orig_forward = module.forward

            def _make_sac_forward(orig_fwd, ctx_fn):
                def _sac_forward(*args, **kwargs):
                    return torch_checkpoint(
                        orig_fwd, *args,
                        use_reentrant=False,
                        context_fn=ctx_fn,
                        **kwargs,
                    )
                return _sac_forward

            module.forward = _make_sac_forward(orig_forward, sac_context_fn)

    return model


class _SACBackend:
    """
    SAC 捕获后端：使用 PyTorch 内置 min_cut_rematerialization_partition，
    同时保存 FW/BW 的 IR 产物用于对比。
    """

    def __init__(self, save_ir: bool = False):
        self.save_ir = save_ir

    def __call__(self, gm, sample_inputs):
        def fw_compiler(gm, _sample_inputs):
            if self.save_ir:
                save_ir_and_dot(gm, model_name=MODEL_NAME,
                                subfolder="partition", graph_name="FW_partitioned")
            return compile_fx_inner(gm, _sample_inputs)

        def bw_compiler(gm, _sample_inputs):
            if self.save_ir:
                save_ir_and_dot(gm, model_name=MODEL_NAME,
                                subfolder="partition", graph_name="BW_partitioned")
            return compile_fx_inner(gm, _sample_inputs, is_backward=True)

        fake_mode = detect_fake_mode(sample_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

        with V.set_fake_mode(fake_mode):
            return aot_module_simplified(
                gm, sample_inputs,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
                decompositions=select_decomp_table(),
            )


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _print_artifacts(base_dir):
    """列出输出文件。"""
    print(f"\n{_BOLD}")
    print(f"  IR 文件已保存至: {base_dir}/")

    for sub in ["joint_graph", "partition"]:
        sub_dir = os.path.join(base_dir, sub)
        if os.path.isdir(sub_dir):
            files = sorted(os.listdir(sub_dir))
            print(f"\n  {sub}/")
            for f in files:
                size = os.path.getsize(os.path.join(sub_dir, f))
                unit = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                print(f"    {f:40s} {unit}")

    print(f"\n{_BOLD}\n")


def capture_aten_ir(strategy_config):
    """使用 ATenIR 自定义 partition_fn 捕获。"""
    model = SimpleMLP(input_dim=64, hidden_dim=128, output_dim=10)
    model.to(device).train()

    print(f"  策略: {strategy_config}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(_LINE)

    layer_pairs = [(model.fc1, 0), (model.fc2, 1)]
    inject_layer_tags(layer_pairs)
    print(f"  已注入层级标签: {len(layer_pairs)} 层")

    print(f"\n  编译中...")
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    return compiled


def capture_sac():
    """使用 PyTorch 内置 SAC (min-cut) 捕获。"""
    model = SimpleMLP(input_dim=64, hidden_dim=128, output_dim=10)
    model.to(device).train()

    print(f"  策略: PyTorch SAC (min_cut_rematerialization_partition)")
    print(f"  SAC Policy: MUST_SAVE={{{', '.join(str(op).split('.')[-2] for op in _COMPUTE_OPS)}}}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(_LINE)

    _wrap_model_with_sac(model)
    print(f"  已包装 SAC checkpoint")

    print(f"\n  编译中...")
    backend = _SACBackend(save_ir=True)
    compiled = torch.compile(model, backend=backend, dynamic=True)
    return compiled


def main():
    strategy_config = json.loads(os.getenv("RECOMPUTE", '{"6": 0}'))
    is_sac = "sac" in strategy_config

    print(f"\n{_BOLD}")
    print(f"  SimpleMLP IR Capture {'(PyTorch SAC)' if is_sac else ''}")
    print(f"  设备: {device}")
    print(_BOLD)

    if is_sac:
        compiled = capture_sac()
    else:
        compiled = capture_aten_ir(strategy_config)

    # ── 触发编译：前向 + 反向 ─────────────────────────────────────────
    x = torch.randn(4, 64, device=device)
    target = torch.randint(0, 10, (4,), device=device)

    output = compiled(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    print(f"  loss = {loss.item():.4f}")

    base_dir = os.path.join(
        os.getenv("PROJECT_ROOT", os.getcwd()),
        "IR_artifacts", MODEL_NAME,
    )
    _print_artifacts(base_dir)


if __name__ == "__main__":
    main()
