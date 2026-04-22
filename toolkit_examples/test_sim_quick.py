#!/usr/bin/env python
"""快速仿真精度测试 — 小模型验证 L2/L2.5/L3 修复效果

使用 GPT-2 小模型 (4L/256H)，batch=4, seq=128。
用 Adam(fused=True) 确保峰值在 FW/BW 阶段而非 OPT。

策略:
  1. Baseline (inductor, no AC)
  2. AC + inductor
  3. inductor budget=0.0 (extreme min-cut)
"""

import gc
import sys
from pathlib import Path

import torch
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.utils import setup_experiment_env, format_bytes
setup_experiment_env()

from toolkit.capture import capture_graphs, capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.profiler import measure_phased, validate
from toolkit.simulation import (
    estimate_inductor_training_peak,
    estimate_training_peak,
    make_level_stub,
    detect_recomputation,
)
from toolkit.strategy import (
    clear_memory_budget,
    get_partition_fn,
    set_memory_budget,
    wrap_with_checkpoint,
)

DEVICE = "cuda"
MODEL_NAME = "gpt2"
MODEL_OVERRIDES = dict(n_layer=4, n_embd=256, n_head=4, n_inner=1024)
BATCH = 4
SEQ = 128
OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3, fused=True)
FUSED_OPTIMIZER = True


def _cleanup():
    torch._dynamo.reset()
    clear_memory_budget()
    gc.collect()
    torch.cuda.empty_cache()


def _mre(est, rt):
    if rt == 0:
        return 0.0
    return abs(est - rt) / rt * 100


def _dir(est, rt):
    if est >= rt:
        return "over"
    return "under"


def run_one(name, registry, input_ids, block_cls, *,
            use_ac=False, budget=None):
    """Run a single strategy: capture → simulate → measure → compare."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    _cleanup()

    # ── 1. Runtime measurement ──
    model = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    if use_ac:
        wrap_with_checkpoint(model, block_cls)
    if budget is not None:
        set_memory_budget(budget)

    compiled = torch.compile(model, backend="inductor", dynamic=True)
    opt = OPTIMIZER_CLS(model.parameters(), **OPTIMIZER_KWARGS)
    forward_fn = lambda: compiled(input_ids=input_ids, labels=input_ids).loss

    rt = measure_phased(name, forward_fn, opt, repeats=3, warmup=2, device=DEVICE)
    del compiled, opt
    _cleanup()

    # ── 2. Inductor capture + L2/L2.5/L3 simulation ──
    model2 = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    if use_ac:
        wrap_with_checkpoint(model2, block_cls)

    has_recomp_override = True if (use_ac or (budget is not None and budget < 0.5)) else None

    cap = capture_inductor_graphs(
        model2, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
        budget=budget,
    )

    sim = estimate_inductor_training_peak(
        cap, model2,
        optimizer_cls=OPTIMIZER_CLS,
        fused_optimizer=FUSED_OPTIMIZER,
        has_recomputation=has_recomp_override,
    )

    del model2
    _cleanup()

    # ── 3. Print results ──
    print(f"\n  Runtime:")
    print(f"    fw_peak  = {format_bytes(rt.fw_peak):>10s}")
    print(f"    bw_peak  = {format_bytes(rt.bw_peak):>10s}")
    print(f"    opt_peak = {format_bytes(rt.opt_peak):>10s}")
    print(f"    true_peak= {format_bytes(rt.overall_peak):>10s}  (phase={rt.peak_phase})")
    print(f"    step_ms  = {rt.step_ms:.1f} ms")

    print(f"\n  {'Level':<6s} {'true_peak':>12s} {'MRE':>8s} {'Dir':>6s}  {'peak_phase':>10s}")
    print(f"  {'-'*50}")

    rows = []
    for level, prefix in [("L2", None), ("L2.5", "l25"), ("L3", "l3")]:
        if prefix is None:
            est_peak = sim["true_peak"]
            phase = sim["peak_phase"]
        else:
            stub = make_level_stub(sim, prefix)
            if stub is None:
                print(f"  {level:<6s} {'N/A':>12s}")
                continue
            est_peak = stub["true_peak"]
            phase = stub.get("peak_phase", "?")

        mre = _mre(est_peak, rt.overall_peak)
        d = _dir(est_peak, rt.overall_peak)
        print(f"  {level:<6s} {format_bytes(est_peak):>12s} {mre:>7.1f}% {d:>6s}  {phase:>10s}")
        rows.append((level, est_peak, mre, d, phase))

    # Extra info
    has_recomp = sim.get("has_recomputation", False)
    print(f"\n  has_recomputation = {has_recomp}")
    if "fusion_fw_eliminated_bytes" in sim:
        print(f"  fusion FW eliminated = {format_bytes(sim['fusion_fw_eliminated_bytes'])}")
        print(f"  fusion BW eliminated = {format_bytes(sim['fusion_bw_eliminated_bytes'])}")

    return rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required")

    registry = ModelRegistry()
    config = registry.get_config(MODEL_NAME, **MODEL_OVERRIDES)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)
    block_cls = registry.get_block_class_name(MODEL_NAME)

    print(f"Model: {MODEL_NAME} {MODEL_OVERRIDES}")
    print(f"Batch={BATCH}, Seq={SEQ}, Optimizer=Adam(fused=True)")

    all_results = {}

    # Strategy 1: Baseline (inductor, no AC)
    all_results["baseline"] = run_one(
        "S1: Inductor Baseline", registry, input_ids, block_cls,
    )

    # Strategy 2: AC + inductor
    all_results["ac"] = run_one(
        "S2: AC + Inductor", registry, input_ids, block_cls,
        use_ac=True,
    )

    # Strategy 3: inductor budget=0.0 (extreme min-cut)
    all_results["budget0"] = run_one(
        "S3: Inductor budget=0.0", registry, input_ids, block_cls,
        budget=0.0,
    )

    # ── Summary table ──
    print(f"\n\n{'='*70}")
    print(f"  Summary: MRE (%) by Level × Strategy")
    print(f"{'='*70}")
    print(f"  {'Strategy':<25s} {'L2':>8s} {'L2.5':>8s} {'L3':>8s}")
    print(f"  {'-'*55}")
    for sname, rows in all_results.items():
        mres = {r[0]: r[2] for r in rows}
        l2 = f"{mres.get('L2', -1):.1f}%" if 'L2' in mres else "N/A"
        l25 = f"{mres.get('L2.5', -1):.1f}%" if 'L2.5' in mres else "N/A"
        l3 = f"{mres.get('L3', -1):.1f}%" if 'L3' in mres else "N/A"
        print(f"  {sname:<25s} {l2:>8s} {l25:>8s} {l3:>8s}")
    print()


if __name__ == "__main__":
    main()
