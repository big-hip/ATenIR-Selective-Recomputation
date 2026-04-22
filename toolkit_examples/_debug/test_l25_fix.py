#!/usr/bin/env python
"""L2.5 修复诊断脚本 — 对比 4 种估算方式的精度

跑一个小规模 LLaMA 模型 + 2 种 Inductor 策略 + ac+inductor,
比较:
  - L2:        标准图级估算 (fusion_aware=False, optimize_order=False)
  - L2.5cur:   当前 L2.5   (fusion_aware=True,  optimize_order=False) + min(sched_bw)
  - L2.5oo:    order_only   (fusion_aware=False, optimize_order=True)
  - L2.5fix:   修复后 L2.5 (fusion_aware=True,  optimize_order=True)  无 Scheduler 混入
  - L3:        Scheduler 级估算

输出: 每策略的 RT / L2 / L2.5cur / L2.5fix / L3 峰值和 MRE
"""

import gc
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from toolkit.models import ModelRegistry
from toolkit.capture import capture_inductor_graphs
from toolkit.simulation import estimate_graph_peak, estimate_training_peak
from toolkit.profiler import measure_phased
from toolkit.strategy import (
    set_memory_budget, clear_memory_budget,
    wrap_with_checkpoint,
)
from toolkit.utils import count_unique_params, format_bytes


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


DEVICE = "cuda"
BATCH, SEQ = 4, 512
OPT_CLS = torch.optim.Adam
OPT_KWARGS = {"lr": 1e-3}
FUSED = False
MODEL_NAME = "llama"


def run_one(budget, label, use_ac=False):
    """Run one strategy and compare all estimation levels."""
    print(f"\n{'='*70}")
    print(f"  {label}  (budget={budget}, ac={use_ac})")
    print(f"{'='*70}")

    _cleanup()

    reg = ModelRegistry()
    cfg = reg.get_config(MODEL_NAME)
    block_cls = reg.get_block_class_name(MODEL_NAME)
    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device=DEVICE)

    # ── Runtime measurement ──
    m = reg.create_model(MODEL_NAME).to(DEVICE).train()
    if use_ac:
        wrap_with_checkpoint(m, block_cls)

    if budget is not None:
        set_memory_budget(budget)
    torch._dynamo.reset()
    compiled = torch.compile(m, backend="inductor", dynamic=True)

    opt = OPT_CLS(m.parameters(), **OPT_KWARGS)
    forward_fn = lambda: compiled(input_ids=input_ids, labels=input_ids).loss

    _cleanup()
    rt = measure_phased(label, forward_fn, opt, repeats=3, warmup=2, device=DEVICE)
    rt_peak = rt.overall_peak
    print(f"  RT peak = {format_bytes(rt_peak)}")
    print(f"  RT phases: FW={format_bytes(rt.fw_peak)} BW={format_bytes(rt.bw_peak)} OPT={format_bytes(rt.opt_peak)} [{rt.peak_phase}]")

    del compiled, opt, m
    if budget is not None:
        clear_memory_budget()
    torch._dynamo.reset()
    _cleanup()

    # ── Capture inductor graphs ──
    m2 = reg.create_model(MODEL_NAME).to(DEVICE).train()
    if use_ac:
        wrap_with_checkpoint(m2, block_cls)

    cap = capture_inductor_graphs(
        m2, input_ids,
        loss_fn=lambda out: out.logits.sum(),
        model_kwargs={"labels": input_ids},
        budget=budget,
    )
    fw_gm = cap["fw_gm"]
    bw_gm = cap["bw_gm"]
    sched_fw = cap.get("sched_fw_peak")
    sched_bw = cap.get("sched_bw_peak")
    print(f"  sched_fw = {format_bytes(sched_fw) if sched_fw else 'N/A'}")
    print(f"  sched_bw = {format_bytes(sched_bw) if sched_bw else 'N/A'}")

    # ── L2: standard graph-level ──
    l2 = estimate_training_peak(fw_gm, bw_gm, m2,
                                 optimizer_cls=OPT_CLS, fused_optimizer=FUSED)
    static_base = l2["base"]
    grad_bytes = l2["grad_bytes"]
    opt_temp = l2["opt_temp"]
    l2_peak = l2["true_peak"]

    # ── L2.5 current: fusion_aware only + min(sched_bw) ──
    fw_fa = estimate_graph_peak(fw_gm, pin_output_inputs=True, fusion_aware=True)
    bw_fa = estimate_graph_peak(bw_gm, pin_output_inputs=True, fusion_aware=True)
    cur_fw = static_base + fw_fa["peak_bytes"]
    if sched_bw is not None:
        cur_bw = static_base + min(sched_bw, bw_fa["peak_bytes"])
    else:
        cur_bw = static_base + bw_fa["peak_bytes"]
    cur_opt = static_base + grad_bytes + opt_temp
    cur_peak = max(cur_fw, cur_bw, cur_opt)

    # ── L2.5 fix: fusion_aware + simulate_inplace ──
    fw_fo = estimate_graph_peak(fw_gm, pin_output_inputs=True,
                                fusion_aware=True, simulate_inplace=True)
    bw_fo = estimate_graph_peak(bw_gm, pin_output_inputs=True,
                                fusion_aware=True, simulate_inplace=True)
    fix_fw = static_base + fw_fo["peak_bytes"]
    fix_bw = static_base + bw_fo["peak_bytes"]
    fix_opt = static_base + grad_bytes + opt_temp
    fix_peak = max(fix_fw, fix_bw, fix_opt)

    # ── L3: Scheduler-level ──
    if sched_fw is not None and sched_bw is not None:
        l3_fw = static_base + sched_fw
        l3_bw = static_base + sched_bw
        l3_opt = static_base + grad_bytes + opt_temp
        l3_peak = max(l3_fw, l3_bw, l3_opt)
    else:
        l3_peak = None

    # ── Bonus: simulate_inplace only (no fusion) ──
    fw_oo = estimate_graph_peak(fw_gm, pin_output_inputs=True,
                                fusion_aware=False, simulate_inplace=True)
    bw_oo = estimate_graph_peak(bw_gm, pin_output_inputs=True,
                                fusion_aware=False, simulate_inplace=True)
    oo_fw = static_base + fw_oo["peak_bytes"]
    oo_bw = static_base + bw_oo["peak_bytes"]
    oo_opt = static_base + grad_bytes + opt_temp
    oo_peak = max(oo_fw, oo_bw, oo_opt)

    def mre(est, ref):
        return abs(est - ref) / ref * 100 if ref > 0 else float("inf")

    def direction(est, ref):
        return "over" if est > ref else "under"

    # ── Print results ──
    print()
    print(f"  {'Level':<18} {'Peak':>12} {'MRE':>8} {'Dir':>6}  FW/BW/OPT breakdown")
    print(f"  {'-'*75}")
    print(f"  {'RT':<18} {format_bytes(rt_peak):>12}")

    for name, pk, fw_p, bw_p, opt_p in [
        ("L2",             l2_peak,  l2["fw_peak"],  l2["bw_peak"],  l2["opt_peak"]),
        ("L2.5-cur",       cur_peak, cur_fw,         cur_bw,         cur_opt),
        ("L2.5-inplace",  oo_peak,  oo_fw,          oo_bw,          oo_opt),
        ("L2.5-fix",       fix_peak, fix_fw,         fix_bw,         fix_opt),
        ("L3",             l3_peak,  l3_fw if l3_peak else 0,
                                     l3_bw if l3_peak else 0,
                                     l3_opt if l3_peak else 0),
    ]:
        if pk is None:
            print(f"  {name:<18} {'N/A':>12}")
            continue
        m_ = mre(pk, rt_peak)
        d_ = direction(pk, rt_peak)
        phase = "FW" if pk == fw_p else ("BW" if pk == bw_p else "OPT")
        print(f"  {name:<18} {format_bytes(pk):>12} {m_:>7.1f}% {d_:>6}  "
              f"FW={format_bytes(fw_p)} BW={format_bytes(bw_p)} OPT={format_bytes(opt_p)} [{phase}]")

    # Also print fusion stats
    print(f"\n  Fusion stats (FW): groups={fw_fa.get('fusion_groups',0)}, "
          f"internal={fw_fa.get('internal_nodes',0)}, "
          f"eliminated={format_bytes(fw_fa.get('internal_bytes',0))}")
    print(f"  Fusion stats (BW): groups={bw_fa.get('fusion_groups',0)}, "
          f"internal={bw_fa.get('internal_nodes',0)}, "
          f"eliminated={format_bytes(bw_fa.get('internal_bytes',0))}")

    # Print graph peak details
    print(f"\n  Graph-only peaks (no static_base):")
    print(f"    L2:             FW={format_bytes(l2['fw_graph_peak'])} BW={format_bytes(l2['bw_graph_peak'])}")
    print(f"    fusion-aware:   FW={format_bytes(fw_fa['peak_bytes'])} BW={format_bytes(bw_fa['peak_bytes'])}")
    print(f"    inplace-only:   FW={format_bytes(fw_oo['peak_bytes'])} BW={format_bytes(bw_oo['peak_bytes'])} reuses={fw_oo.get('n_reuses',0)}/{bw_oo.get('n_reuses',0)}")
    print(f"    fusion+inplace: FW={format_bytes(fw_fo['peak_bytes'])} BW={format_bytes(bw_fo['peak_bytes'])} reuses={fw_fo.get('n_reuses',0)}/{bw_fo.get('n_reuses',0)}")
    if sched_fw is not None:
        print(f"    L3 scheduler:   FW={format_bytes(sched_fw)} BW={format_bytes(sched_bw)}")

    del m2
    _cleanup()

    return {
        "label": label,
        "rt": rt_peak,
        "l2": l2_peak,
        "l25_cur": cur_peak,
        "l25_oo": oo_peak,
        "l25_fix": fix_peak,
        "l3": l3_peak,
    }


def main():
    torch.manual_seed(42)

    results = []
    for budget, label, use_ac in [
        (1.0,  "inductor(b=1.0)", False),
        (0.0,  "inductor(b=0.0)", False),
        (None, "ac+inductor",     True),
    ]:
        r = run_one(budget, label, use_ac=use_ac)
        results.append(r)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<22} {'RT':>10} {'L2 MRE':>8} {'L25cur':>8} {'L25inp':>8} {'L25fix':>8} {'L3 MRE':>8}")
    print(f"  {'-'*75}")
    for r in results:
        def m(v):
            return f"{abs(v-r['rt'])/r['rt']*100:.1f}%" if v else "N/A"
        print(f"  {r['label']:<22} {format_bytes(r['rt']):>10} "
              f"{m(r['l2']):>8} {m(r['l25_cur']):>8} {m(r['l25_oo']):>8} "
              f"{m(r['l25_fix']):>8} {m(r['l3']):>8}")


if __name__ == "__main__":
    main()
