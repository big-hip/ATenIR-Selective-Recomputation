"""
diag_compile.py

诊断脚本：定位 torch.compile + 自定义 CompilerBackend 导致梯度消失的根因。
测试三种编译模式，比较梯度范数和 loss 变化。

用法:
    cd examples/transformer
    python diag_compile.py
"""
import copy
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from model import Transformer, device
from aten_recompute.core import CompilerBackend, inject_layer_tags

torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64

_BOLD = "═" * 72
_LINE = "─" * 72

# ── 小模型 + 少量数据，快速诊断 ─────────────────────────────────────────────

SRC_VOCAB = 5000
TGT_VOCAB = 5000
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 1024
MAX_SEQ = 32
DROPOUT = 0.1
BATCH = 8
N_STEPS = 5


def make_model():
    m = Transformer(SRC_VOCAB, TGT_VOCAB, D_MODEL, NUM_HEADS,
                    NUM_LAYERS, D_FF, MAX_SEQ, DROPOUT, padding_idx=0)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return m


def make_data():
    src = torch.randint(1, SRC_VOCAB, (BATCH, MAX_SEQ)).to(device)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, MAX_SEQ)).to(device)
    return src, tgt


def grad_norm(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            count += 1
    return total ** 0.5, count


def run_test(tag, model, n_steps=N_STEPS):
    """运行 n_steps 步训练，打印 loss 和梯度范数。"""
    print(f"\n  [{tag}]")
    print(f"  {'Step':>4}  {'Loss':>10}  {'Grad Norm':>12}  {'Params w/ grad':>16}")
    print(f"  {_LINE}")

    try:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        src, tgt = make_data()

        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, TGT_VOCAB),
                tgt[:, 1:].contiguous().view(-1),
            )
            loss.backward()

            gn, gc = grad_norm(model)
            print(f"  {step:4d}  {loss.item():10.4f}  {gn:12.6f}  {gc:16d}")
            losses.append(loss.item())

            optimizer.step()

        delta = losses[0] - losses[-1]
        status = "OK (loss 在下降)" if delta > 0.05 else "FAIL (loss 不降)"
        print(f"  结果: loss {losses[0]:.4f} → {losses[-1]:.4f}, delta={delta:.4f} → {status}")
        return delta > 0.05
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


def main():
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "WARNING")
    # 开启自定义 partition 的诊断日志
    os.environ["RECOMPUTE_DEBUG"] = "1"
    torch.manual_seed(42)

    print(f"\n{_BOLD}")
    print("  Compile 诊断: 梯度流检测")
    print(_BOLD)
    print(f"  设备: {device}")
    print(f"  模型: {NUM_LAYERS}L-{D_MODEL}d (小模型快速测试)")
    print(f"  每组测试: {N_STEPS} 步")
    print(_LINE)

    results = {}

    # ── Test 1: Eager baseline ─────────────────────────────────────────────
    torch.manual_seed(42)
    eager_model = make_model().to(device)
    eager_model.train()
    results["eager"] = run_test("Eager (无编译)", eager_model)
    del eager_model
    torch._dynamo.reset()

    # ── Test 2: torch.compile + 默认 Inductor (无自定义 partition) ──────────
    torch.manual_seed(42)
    default_model = make_model().to(device)
    default_model.train()
    compiled_default = torch.compile(default_model, dynamic=False)
    results["inductor_default"] = run_test("torch.compile 默认 Inductor", compiled_default)
    del default_model, compiled_default
    torch._dynamo.reset()

    # ── Test 2b: CompilerBackend + PASSTHROUGH (默认 partition) ─────────────
    # 关键诊断：若此测试通过但 Test 3 失败，问题在 partition_fn 逻辑；
    # 若此测试也失败，问题在 CompilerBackend 的 aot_module_simplified 调用方式。
    torch.manual_seed(42)
    passthrough_model = make_model().to(device)
    passthrough_model.train()
    enc = [(l, i) for i, l in enumerate(passthrough_model.encoder_layers)]
    dec = [(l, len(passthrough_model.encoder_layers) + i)
           for i, l in enumerate(passthrough_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    os.environ["RECOMPUTE_PASSTHROUGH"] = "1"
    backend_pt = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_pt = torch.compile(passthrough_model, backend=backend_pt, dynamic=False)
    results["passthrough"] = run_test("CompilerBackend + 默认 partition (PASSTHROUGH)", compiled_pt)
    del passthrough_model, compiled_pt
    os.environ.pop("RECOMPUTE_PASSTHROUGH", None)
    torch._dynamo.reset()

    # ── Test 2c: CompilerBackend + 策略 0 但不注入 mark_layer ──────────────
    # 关键诊断：若此测试通过但 Test 3 失败，问题在 mark_layer 清理逻辑。
    torch.manual_seed(42)
    nomark_model = make_model().to(device)
    nomark_model.train()
    # 注意：不调用 inject_layer_tags！
    backend_nomark = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_nomark = torch.compile(nomark_model, backend=backend_nomark, dynamic=False)
    results["strategy_0_no_mark"] = run_test(
        "CompilerBackend 策略 0 (无 mark_layer)", compiled_nomark
    )
    del nomark_model, compiled_nomark
    torch._dynamo.reset()

    # ── Test 3: CompilerBackend + 策略 0 (不重计算) ────────────────────────
    torch.manual_seed(42)
    strat0_model = make_model().to(device)
    strat0_model.train()
    enc = [(l, i) for i, l in enumerate(strat0_model.encoder_layers)]
    dec = [(l, len(strat0_model.encoder_layers) + i)
           for i, l in enumerate(strat0_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend0 = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_s0 = torch.compile(strat0_model, backend=backend0, dynamic=False)
    results["strategy_0"] = run_test("CompilerBackend 策略 0 (不重计算)", compiled_s0)
    del strat0_model, compiled_s0
    torch._dynamo.reset()

    # ── Test 4: CompilerBackend + 策略 6 (廉价重计算) ──────────────────────
    torch.manual_seed(42)
    strat6_model = make_model().to(device)
    strat6_model.train()
    enc = [(l, i) for i, l in enumerate(strat6_model.encoder_layers)]
    dec = [(l, len(strat6_model.encoder_layers) + i)
           for i, l in enumerate(strat6_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend6 = CompilerBackend(strategy_config={"6": 0}, save_ir=False)
    compiled_s6 = torch.compile(strat6_model, backend=backend6, dynamic=False)
    results["strategy_6"] = run_test("CompilerBackend 策略 6 (廉价重计算)", compiled_s6)
    del strat6_model, compiled_s6
    torch._dynamo.reset()

    # ── Test 5: CompilerBackend + 策略 1 (全部重计算) ──────────────────────
    torch.manual_seed(42)
    strat1_model = make_model().to(device)
    strat1_model.train()
    enc = [(l, i) for i, l in enumerate(strat1_model.encoder_layers)]
    dec = [(l, len(strat1_model.encoder_layers) + i)
           for i, l in enumerate(strat1_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend1 = CompilerBackend(strategy_config={"1": None}, save_ir=False)
    compiled_s1 = torch.compile(strat1_model, backend=backend1, dynamic=False)
    results["strategy_1"] = run_test("CompilerBackend 策略 1 (全部重计算)", compiled_s1)
    del strat1_model, compiled_s1
    torch._dynamo.reset()

    # ── Test 6: CompilerBackend + 策略 7 (min-cut 最优) ───────────────────
    torch.manual_seed(42)
    strat7_model = make_model().to(device)
    strat7_model.train()
    enc = [(l, i) for i, l in enumerate(strat7_model.encoder_layers)]
    dec = [(l, len(strat7_model.encoder_layers) + i)
           for i, l in enumerate(strat7_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend7 = CompilerBackend(strategy_config={"7": 0.5}, save_ir=False)
    compiled_s7 = torch.compile(strat7_model, backend=backend7, dynamic=False)
    results["strategy_7"] = run_test("CompilerBackend 策略 7 (min-cut)", compiled_s7)
    del strat7_model, compiled_s7
    torch._dynamo.reset()

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}")
    print("  诊断汇总")
    print(_BOLD)
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name:30s} → {status}")
    print(_LINE)

    # ── 分层诊断分析 ──────────────────────────────────────────────────────
    print()
    if not results.get("eager"):
        print("  诊断: Eager baseline 都失败了，请检查模型和数据！")
    elif not results.get("inductor_default"):
        print("  诊断: 默认 Inductor 编译也失败了，可能是 PyTorch/GPU 环境问题。")
    elif not results.get("passthrough"):
        print("  诊断: PASSTHROUGH 模式也失败。问题在 CompilerBackend 的")
        print("         aot_module_simplified 调用方式（fake_mode / decompositions），")
        print("         而非自定义 partition_fn 逻辑。")
    elif not results.get("strategy_0_no_mark"):
        print("  诊断: 自定义 partition（无 mark_layer）也失败。问题在")
        print("         saved_values 收集逻辑 或 _recursive_joint_graph_passes 副作用。")
    elif not results.get("strategy_0"):
        print("  诊断: mark_layer 注入后失败。问题在 _cleanup_mark_layer 清理逻辑。")
        print("         请检查上方 [cleanup] 日志中是否有悬空引用或去重问题。")
    elif results.get("strategy_0") and not results.get("strategy_6"):
        print("  诊断: 策略 0 正常但策略 6 失败。问题在 _apply_strategy 或")
        print("         _find_required_primals 逻辑。")
    elif all(results.values()):
        print("  诊断: 所有测试通过! 问题可能在长训练/特定数据上才出现。")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  诊断: 失败的测试: {failed}")
        print("         请查看上方详细输出和 RECOMPUTE_DEBUG 日志。")

    print(f"\n{_BOLD}\n")


if __name__ == "__main__":
    main()
