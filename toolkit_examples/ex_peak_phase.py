#!/usr/bin/env python
"""实验 2 — 峰值位置分析: Batch Size × Optimizer 效应

论文 §5.2: 说清楚 Adam/Fused Adam/SGD 和 Batch Size 对峰值阶段的影响。

核心发现:
  1. Adam(non-fused) 仅在小 batch (B=2) 时 peak 在 OPT, B≥4 后 peak 由 FW 主导
  2. Adam(fused) opt_peak 降低约 3~4% (大模型中 optimizer 临时变量占比小)
  3. SGD opt_peak 远低于 Adam → peak 始终在 FW
  4. Batch 增大 → activation 增长, peak 稳定在 FW (grad_bytes 恒定)
  5. B=16 + Adam/Adam(fused) 非 AC 策略 OOM (48GB GPU)

模型:  LLaMA 16L/2048H/16heads/8kv/5504I (~870M params)
Seq:   512

参数矩阵:
  Batch:     [2, 4, 8, 16]
  Optimizer: SGD, Adam, Adam(fused=True)
  策略:      eager, inductor(b=1.0), inductor(b=0.0), ac+inductor

输出:
  - 每 cell: fw_peak, bw_peak, opt_peak, true_peak, peak_phase, step_ms
  - CSV: outputs/ex_peak_phase.csv
"""

import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.utils import setup_experiment_env, count_unique_params, format_bytes
setup_experiment_env()

from toolkit.models import ModelRegistry
from toolkit.output import print_comparison_table, to_csv
from toolkit.profiler import measure_phased
from toolkit.strategy import (
    clear_memory_budget,
    set_memory_budget,
    wrap_with_checkpoint,
)

DEVICE = "cuda"
MODEL_NAME = "llama"
MODEL_OVERRIDES = dict(
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=5504,
    num_key_value_heads=8,
    max_position_embeddings=1024,
)
BATCHES = [2, 4, 8, 16]
SEQ = 512
OUTPUT_DIR = Path(__file__).with_name("outputs")

# ── Optimizer configurations ──
OPTIMIZERS = [
    ("SGD",         lambda p: torch.optim.SGD(p, lr=1e-3)),
    ("Adam",        lambda p: torch.optim.Adam(p, lr=1e-3, fused=False)),
    ("Adam(fused)", lambda p: torch.optim.Adam(p, lr=1e-3, fused=True)),
]

# ── Strategy definitions ──
# (name, wrapping, use_inductor, budget)
STRATEGIES = [
    ("eager",           None, False, None),
    ("inductor(b=1.0)", None, True,  1.0),
    ("inductor(b=0.0)", None, True,  0.0),
    ("ac+inductor",     "ac", True,  None),
]


def _new_model(registry):
    return registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()


def _compile_inductor(model, budget=None):
    if budget is not None:
        set_memory_budget(budget)
    torch._dynamo.reset()
    return torch.compile(model, backend="inductor", dynamic=True)


def _measure(label, forward_fn, optimizer):
    return measure_phased(label, forward_fn, optimizer, repeats=3, warmup=2, device=DEVICE)


def _cleanup():
    torch._dynamo.reset()
    clear_memory_budget()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    block_cls = registry.get_block_class_name(MODEL_NAME)

    tmp = _new_model(registry)
    param_bytes = count_unique_params(tmp)
    del tmp
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 80)
    print("  实验 2: 峰值位置分析 — Batch × Optimizer 效应")
    print(f"  Model: {MODEL_NAME} ({MODEL_OVERRIDES['num_hidden_layers']}L/"
          f"{MODEL_OVERRIDES['hidden_size']}H), ~{param_bytes / 1e6:.0f}M params")
    print(f"  Batches: {BATCHES} | Seq: {SEQ}")
    print(f"  Optimizers: {[n for n, _ in OPTIMIZERS]}")
    print(f"  Strategies: {[s[0] for s in STRATEGIES]}")
    print(f"  param_bytes = {format_bytes(param_bytes)}")
    print("=" * 80)

    all_rows = []

    for opt_name, opt_factory in OPTIMIZERS:
        print(f"\n  ── Optimizer: {opt_name} ──")

        for batch in BATCHES:
            config = registry.get_config(MODEL_NAME, **MODEL_OVERRIDES)
            input_ids = torch.randint(0, config.vocab_size, (batch, SEQ), device=DEVICE)

            for strat_name, wrapping, use_inductor, budget in STRATEGIES:
                label = f"{opt_name}/{strat_name}/B={batch}"
                print(f"    {label:50s} ...", end=" ", flush=True)
                try:
                    gc.collect()
                    torch.cuda.empty_cache()

                    m = _new_model(registry)
                    if wrapping == "ac":
                        wrap_with_checkpoint(m, block_cls)

                    if use_inductor:
                        compiled = _compile_inductor(m, budget=budget)
                    else:
                        compiled = m

                    opt = opt_factory(m.parameters())
                    rt = _measure(
                        label,
                        lambda c=compiled, ids=input_ids: c(input_ids=ids, labels=ids).loss,
                        opt,
                    )

                    row = {
                        "optimizer": opt_name,
                        "strategy": strat_name,
                        "batch": batch,
                        "fw_peak": rt.fw_peak,
                        "bw_peak": rt.bw_peak,
                        "opt_peak": rt.opt_peak,
                        "fwbw_peak": rt.fwbw_peak,
                        "true_peak": rt.overall_peak,
                        "peak_phase": rt.peak_phase,
                        "step_ms": round(rt.step_ms, 1),
                        "base": rt.base_allocated,
                    }
                    all_rows.append(row)

                    print(f"peak={format_bytes(rt.overall_peak):>9s}  "
                          f"phase={rt.peak_phase:>3s}  "
                          f"fw={format_bytes(rt.fw_peak):>9s}  "
                          f"bw={format_bytes(rt.bw_peak):>9s}  "
                          f"opt={format_bytes(rt.opt_peak):>9s}  "
                          f"step={rt.step_ms:.1f}ms")

                    del m, compiled, opt
                    _cleanup()

                except Exception as e:
                    print(f"FAILED ({e})")
                    import traceback
                    traceback.print_exc()
                    # Record OOM / failure row so CSV stays complete
                    all_rows.append({
                        "optimizer": opt_name,
                        "strategy": strat_name,
                        "batch": batch,
                        "fw_peak": None, "bw_peak": None, "opt_peak": None,
                        "fwbw_peak": None, "true_peak": None,
                        "peak_phase": "OOM",
                        "step_ms": None, "base": None,
                    })
                    _cleanup()

    # ══════════════════════════════════════════════════════════════════
    #  Output
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)

    # T1: Full table
    summary = []
    for r in all_rows:
        summary.append({
            "optimizer": r["optimizer"],
            "batch": r["batch"],
            "strategy": r["strategy"],
            "true_peak": format_bytes(r["true_peak"]) if r["true_peak"] else "OOM",
            "peak_phase": r["peak_phase"],
            "fw_peak": format_bytes(r["fw_peak"]) if r["fw_peak"] else "OOM",
            "bw_peak": format_bytes(r["bw_peak"]) if r["bw_peak"] else "OOM",
            "opt_peak": format_bytes(r["opt_peak"]) if r["opt_peak"] else "OOM",
            "step_ms": r["step_ms"],
        })
    print_comparison_table(summary, title="T1: Optimizer × Batch × Strategy — Peak Phase Analysis")

    # T2: Peak phase distribution
    print("\n  T2: Peak Phase Distribution")
    print(f"  {'Optimizer':15s} {'Batch':>5s} {'#FW':>4s} {'#BW':>4s} {'#OPT':>4s}")
    print("  " + "-" * 40)
    for opt_name, _ in OPTIMIZERS:
        for batch in BATCHES:
            batch_rows = [r for r in all_rows
                          if r["optimizer"] == opt_name and r["batch"] == batch]
            n_fw = sum(1 for r in batch_rows if r["peak_phase"] == "FW")
            n_bw = sum(1 for r in batch_rows if r["peak_phase"] == "BW")
            n_opt = sum(1 for r in batch_rows if r["peak_phase"] == "OPT")
            print(f"  {opt_name:15s} {batch:>5d} {n_fw:>4d} {n_bw:>4d} {n_opt:>4d}")

    # T3: Activation ratio analysis (fw_peak vs grad_bytes=param_bytes)
    print(f"\n  T3: Activation Ratio (fw_peak / param_bytes) — eager strategy")
    print(f"  {'Optimizer':15s} {'Batch':>5s} {'fw_peak':>12s} {'grad_bytes':>12s} {'ratio':>8s} {'insight'}")
    print("  " + "-" * 70)
    for opt_name, _ in OPTIMIZERS:
        for batch in BATCHES:
            eager_rows = [r for r in all_rows
                          if r["optimizer"] == opt_name and r["batch"] == batch
                          and r["strategy"] == "eager"]
            if eager_rows and eager_rows[0]["fw_peak"] is not None:
                r = eager_rows[0]
                ratio = r["fw_peak"] / param_bytes if param_bytes > 0 else 0
                insight = ("grad-dominated" if ratio < 1.5
                           else "mixed" if ratio < 3
                           else "act-dominated")
                print(f"  {opt_name:15s} {batch:>5d} "
                      f"{format_bytes(r['fw_peak']):>12s} "
                      f"{format_bytes(param_bytes):>12s} "
                      f"{ratio:>7.1f}x  {insight}")

    # T4: Adam vs fused Adam opt_peak comparison
    print(f"\n  T4: Adam vs Fused Adam — opt_peak")
    print(f"  {'Batch':>5s} {'Strategy':20s} {'Adam opt_peak':>14s} {'Fused opt_peak':>14s} {'Reduction':>10s}")
    print("  " + "-" * 70)
    for batch in BATCHES:
        for strat_name, _, _, _ in STRATEGIES:
            adam_rows = [r for r in all_rows
                         if r["optimizer"] == "Adam" and r["batch"] == batch
                         and r["strategy"] == strat_name]
            fused_rows = [r for r in all_rows
                          if r["optimizer"] == "Adam(fused)" and r["batch"] == batch
                          and r["strategy"] == strat_name]
            if (adam_rows and fused_rows
                    and adam_rows[0]["opt_peak"] is not None
                    and fused_rows[0]["opt_peak"] is not None):
                adam_opt = adam_rows[0]["opt_peak"]
                fused_opt = fused_rows[0]["opt_peak"]
                reduction = (adam_opt - fused_opt) / adam_opt * 100 if adam_opt > 0 else 0
                print(f"  {batch:>5d} {strat_name:20s} "
                      f"{format_bytes(adam_opt):>14s} "
                      f"{format_bytes(fused_opt):>14s} "
                      f"{reduction:>9.1f}%")

    csv_path = OUTPUT_DIR / "ex_peak_phase.csv"
    to_csv(all_rows, csv_path)
    print(f"\n  CSV -> {csv_path}")



if __name__ == "__main__":
    main()
