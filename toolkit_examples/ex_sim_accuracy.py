#!/usr/bin/env python
"""实验 1 — 全策略仿真精度验证

论文 §5.1: 验证 L2/L2.5/L3 仿真在 12 种重计算策略下的精度。

模型:  LLaMA 16L/2048H/16heads/8kv/5504I (~870M params)
Batch: 8, Seq: 512
优化器: Adam(fused=True) — 生产配置

策略分组:
  G1 Eager:      S01 eager, S02 classic_ac, S03 sac_mm, S04 sac_all
  G2 Compiled:   S05 aot_eager+default, S06 aot_eager+min_cut,
                 S07 inductor(b=1.0), S08 inductor(b=0.5), S09 inductor(b=0.0)
  G3 AC+Compiled: S10 ac+aot_eager, S11 ac+inductor, S12 sac_mm+inductor

输出:
  - 每策略: fw_peak/bw_peak/opt_peak/true_peak/peak_phase + L2/L2.5/L3 MRE
  - CSV: outputs/ex_sim_accuracy.csv
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

from toolkit.utils import setup_experiment_env, count_unique_params, format_bytes
setup_experiment_env()

from toolkit.capture import capture_graphs, capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.output import print_comparison_table, to_csv
from toolkit.profiler import measure_phased, validate, analyze_error_sources
from toolkit.simulation import estimate_inductor_training_peak, estimate_training_peak, make_level_stub
from toolkit.strategy import (
    clear_memory_budget,
    get_partition_fn,
    set_memory_budget,
    wrap_with_checkpoint,
    wrap_with_sac,
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
BATCH = 8
SEQ = 512
OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3, fused=True)
FUSED_OPTIMIZER = True
OUTPUT_DIR = Path(__file__).with_name("outputs")


# ── Helpers ───────────────────────────────────────────────────────────
def _new_model(registry):
    return registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()


def _compile_aot_eager(model, partition_fn):
    def backend(gm, example_inputs):
        def fw_compiler(fw_gm, _inputs):
            return make_boxed_func(fw_gm.forward)
        def bw_compiler(bw_gm, _inputs):
            return make_boxed_func(bw_gm.forward)
        return aot_module_simplified(
            gm, example_inputs,
            fw_compiler=fw_compiler, bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
    torch._dynamo.reset()
    return torch.compile(model, backend=backend, dynamic=True)


def _compile_inductor(model, budget=None):
    if budget is not None:
        set_memory_budget(budget)
    torch._dynamo.reset()
    return torch.compile(model, backend="inductor", dynamic=True)


def _measure(label, forward_fn, optimizer):
    return measure_phased(
        label, forward_fn, optimizer,
        repeats=3, warmup=2, device=DEVICE,
    )


def _cleanup():
    torch._dynamo.reset()
    clear_memory_budget()
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
def main():
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    config = registry.get_config(MODEL_NAME, **MODEL_OVERRIDES)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)
    block_cls = registry.get_block_class_name(MODEL_NAME)

    tmp = _new_model(registry)
    param_bytes = count_unique_params(tmp)
    del tmp
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 80)
    print("  实验 1: 全策略仿真精度验证")
    print(f"  Model: {MODEL_NAME} ({MODEL_OVERRIDES['num_hidden_layers']}L/"
          f"{MODEL_OVERRIDES['hidden_size']}H), ~{param_bytes / 1e6:.0f}M params")
    print(f"  Batch={BATCH}, Seq={SEQ}, Optimizer=Adam(fused=True)")
    print(f"  param_bytes = {format_bytes(param_bytes)}")
    print("=" * 80)

    all_rows = []
    summary_rows = []
    phase_rows = []

    # ──────────────────────────────────────────────────────────────────
    #  Helper: run one strategy
    # ──────────────────────────────────────────────────────────────────
    def run_strategy(name, group, make_compiled_fn, partition_name=None,
                      capture_mode=None, budget=None, wrapping=None):
        if partition_name is not None and capture_mode is None:
            capture_mode = "aot_eager"
        print(f"    {name:35s} ...", end=" ", flush=True)
        try:
            gc.collect()
            torch.cuda.empty_cache()

            # ── Runtime measurement ──
            m = _new_model(registry)
            compiled, _ = make_compiled_fn(m)
            opt = OPTIMIZER_CLS(m.parameters(), **OPTIMIZER_KWARGS)
            rt = _measure(name, lambda c=compiled, ids=input_ids: c(input_ids=ids, labels=ids).loss, opt)

            del compiled, opt, m
            _cleanup()

            # ── L2/L3 Simulation ──
            l2 = None
            val = None
            val_l25 = None
            val_l3 = None

            if capture_mode == "aot_eager":
                m_sim = _new_model(registry)
                if wrapping == "ac":
                    wrap_with_checkpoint(m_sim, block_cls)
                elif wrapping == "sac":
                    wrap_with_sac(m_sim, block_cls, "save_matmuls")

                fw_gm, bw_gm = capture_graphs(
                    m_sim, input_ids, lambda out: out.loss,
                    model_kwargs={"labels": input_ids},
                    partition_fn=get_partition_fn(partition_name or "default"),
                )
                l2 = estimate_training_peak(
                    fw_gm, bw_gm, m_sim,
                    optimizer_cls=OPTIMIZER_CLS,
                    fused_optimizer=FUSED_OPTIMIZER,
                )
                val = validate(l2, rt)
                del m_sim, fw_gm, bw_gm
                gc.collect()
                torch.cuda.empty_cache()

            elif capture_mode == "inductor":
                m_sim = _new_model(registry)
                if wrapping == "ac":
                    wrap_with_checkpoint(m_sim, block_cls)
                elif wrapping == "sac":
                    wrap_with_sac(m_sim, block_cls, "save_matmuls")

                cap = capture_inductor_graphs(
                    m_sim, input_ids, lambda out: out.loss,
                    model_kwargs={"labels": input_ids},
                    budget=budget,
                )
                l2 = estimate_inductor_training_peak(
                    cap, m_sim,
                    optimizer_cls=OPTIMIZER_CLS,
                    fused_optimizer=FUSED_OPTIMIZER,
                    has_recomputation=(
                        True if (wrapping in ("ac", "sac")
                                 or (budget is not None and budget < 0.5))
                        else None
                    ),
                )
                val = validate(l2, rt)
                l25_stub = make_level_stub(l2, "l25")
                if l25_stub is not None:
                    val_l25 = validate(l25_stub, rt)
                l3_stub = make_level_stub(l2, "l3")
                if l3_stub is not None:
                    val_l3 = validate(l3_stub, rt)
                del m_sim, cap
                gc.collect()
                torch.cuda.empty_cache()

            # ── Build result row ──
            row = {
                "strategy": name,
                "group": group,
                "rt_fw_peak": rt.fw_peak,
                "rt_bw_peak": rt.bw_peak,
                "rt_opt_peak": rt.opt_peak,
                "rt_fwbw_peak": rt.fwbw_peak,
                "rt_true_peak": rt.overall_peak,
                "rt_peak_phase": rt.peak_phase,
                "rt_step_ms": round(rt.step_ms, 1),
                "rt_base": rt.base_allocated,
            }
            if l2 is not None:
                err = analyze_error_sources(l2, rt)
                row.update({
                    "l2_fw_peak": l2["fw_peak"],
                    "l2_bw_peak": l2["bw_peak"],
                    "l2_opt_peak": l2["opt_peak"],
                    "l2_fwbw_peak": l2["fwbw_peak"],
                    "l2_true_peak": l2["true_peak"],
                    "l2_peak_phase": l2["peak_phase"],
                    "mre": val.mre_allocated,
                    "direction": val.direction,
                    "err_total": err["total_error"],
                    "err_pct": err["total_error_pct"],
                    "err_fixed": err["sources"][0][1],
                    "err_act": err["sources"][1][1],
                    "err_alloc_overhead": err["sources"][2][1],
                    "l25_true_peak": l2.get("l25_true_peak"),
                    "l25_mre": val_l25.mre_allocated if val_l25 else None,
                    "l25_direction": val_l25.direction if val_l25 else None,
                    "l3_true_peak": l2.get("l3_true_peak"),
                    "l3_mre": val_l3.mre_allocated if val_l3 else None,
                    "l3_direction": val_l3.direction if val_l3 else None,
                })
            else:
                row.update({
                    "l2_fw_peak": None, "l2_bw_peak": None, "l2_opt_peak": None,
                    "l2_fwbw_peak": None, "l2_true_peak": None, "l2_peak_phase": None,
                    "mre": None, "direction": None,
                    "err_total": None, "err_pct": None,
                    "err_fixed": None, "err_act": None, "err_alloc_overhead": None,
                    "l25_true_peak": None, "l25_mre": None, "l25_direction": None,
                    "l3_true_peak": None, "l3_mre": None, "l3_direction": None,
                })

            all_rows.append(row)

            # ── Summary row ──
            l2_str = format_bytes(l2["true_peak"]) if l2 else "N/A"
            mre_str = f"{val.mre_allocated * 100:.1f}%" if val else "N/A"
            dir_str = val.direction if val else "N/A"
            l25_str = format_bytes(l2["l25_true_peak"]) if l2 and l2.get("l25_true_peak") else "N/A"
            l25_mre_str = f"{val_l25.mre_allocated * 100:.1f}%" if val_l25 else "N/A"
            l3_str = format_bytes(l2["l3_true_peak"]) if l2 and l2.get("l3_true_peak") else "N/A"
            l3_mre_str = f"{val_l3.mre_allocated * 100:.1f}%" if val_l3 else "N/A"
            summary_rows.append({
                "strategy": name,
                "group": group,
                "L2_peak": l2_str,
                "L2.5_peak": l25_str,
                "L3_peak": l3_str,
                "Runtime_peak": format_bytes(rt.overall_peak),
                "L2_MRE": mre_str,
                "L2.5_MRE": l25_mre_str,
                "L3_MRE": l3_mre_str,
                "direction": dir_str,
                "rt_phase": rt.peak_phase,
                "step_ms": round(rt.step_ms, 1),
            })

            if val and val.breakdown:
                bd = val.breakdown
                phase_rows.append({
                    "strategy": name,
                    "fw_MRE": f"{bd.get('mre_fw_peak', 0) * 100:.1f}%",
                    "bw_MRE": f"{bd.get('mre_bw_peak', 0) * 100:.1f}%",
                    "opt_MRE": f"{bd.get('mre_opt_peak', 0) * 100:.1f}%",
                    "overall_MRE": mre_str,
                })

            print(f"RT={format_bytes(rt.overall_peak):>10s}  "
                  f"L2={l2_str:>10s}  L2.5={l25_str:>10s}  L3={l3_str:>10s}  "
                  f"MRE_L2={mre_str:>6s}  MRE_L2.5={l25_mre_str:>6s}  MRE_L3={l3_mre_str:>6s}")
            return row

        except Exception as e:
            print(f"FAILED ({e})")
            import traceback
            traceback.print_exc()
            _cleanup()
            return None

    # ══════════════════════════════════════════════════════════════════
    #  Group 1: Eager (no graph → L2 = N/A)
    # ══════════════════════════════════════════════════════════════════
    print("\n  G1: Eager (L2 not applicable)")

    def _eager(model):
        return model, model
    run_strategy("S01 eager_baseline", "G1", _eager)

    def _eager_ac(model):
        wrap_with_checkpoint(model, block_cls)
        return model, model
    run_strategy("S02 classic_ac", "G1", _eager_ac)

    def _eager_sac_mm(model):
        wrap_with_sac(model, block_cls, "save_matmuls")
        return model, model
    run_strategy("S03 sac_save_matmuls", "G1", _eager_sac_mm)

    def _eager_sac_all(model):
        wrap_with_sac(model, block_cls, "recompute_all")
        return model, model
    run_strategy("S04 sac_recompute_all", "G1", _eager_sac_all)

    # ══════════════════════════════════════════════════════════════════
    #  Group 2: Compiled
    # ══════════════════════════════════════════════════════════════════
    print("\n  G2: Compiled")

    def _aot_default(model):
        c = _compile_aot_eager(model, get_partition_fn("default"))
        return c, model
    run_strategy("S05 aot_eager+default", "G2", _aot_default, partition_name="default")

    def _aot_mincut(model):
        c = _compile_aot_eager(model, get_partition_fn("min_cut"))
        return c, model
    run_strategy("S06 aot_eager+min_cut", "G2", _aot_mincut, partition_name="min_cut")

    def _ind_b10(model):
        c = _compile_inductor(model, budget=1.0)
        return c, model
    run_strategy("S07 inductor(b=1.0)", "G2", _ind_b10, capture_mode="inductor", budget=1.0)

    def _ind_b05(model):
        c = _compile_inductor(model, budget=0.5)
        return c, model
    run_strategy("S08 inductor(b=0.5)", "G2", _ind_b05, capture_mode="inductor", budget=0.5)

    def _ind_b00(model):
        c = _compile_inductor(model, budget=0.0)
        return c, model
    run_strategy("S09 inductor(b=0.0)", "G2", _ind_b00, capture_mode="inductor", budget=0.0)

    # ══════════════════════════════════════════════════════════════════
    #  Group 3: AC/SAC + Compiled
    # ══════════════════════════════════════════════════════════════════
    print("\n  G3: AC/SAC + Compiled")

    def _ac_aot(model):
        wrap_with_checkpoint(model, block_cls)
        c = _compile_aot_eager(model, get_partition_fn("default"))
        return c, model
    run_strategy("S10 ac+aot_eager(default)", "G3", _ac_aot, partition_name="default", wrapping="ac")

    def _ac_inductor(model):
        wrap_with_checkpoint(model, block_cls)
        c = _compile_inductor(model)
        return c, model
    run_strategy("S11 ac+inductor", "G3", _ac_inductor, capture_mode="inductor", wrapping="ac")

    def _sac_inductor(model):
        wrap_with_sac(model, block_cls, "save_matmuls")
        c = _compile_inductor(model)
        return c, model
    run_strategy("S12 sac_mm+inductor", "G3", _sac_inductor, capture_mode="inductor", wrapping="sac")

    # ══════════════════════════════════════════════════════════════════
    #  Output
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)

    print_comparison_table(summary_rows,
                           title="全策略仿真精度 — L2/L2.5/L3 vs Runtime")

    if phase_rows:
        print_comparison_table(phase_rows,
                               title="分阶段 MRE — FW / BW / OPT")

    # ── Accuracy summary ──
    sim_rows = [r for r in all_rows if r["mre"] is not None]
    if sim_rows:
        avg_mre = sum(r["mre"] for r in sim_rows) / len(sim_rows) * 100
        max_mre = max(r["mre"] for r in sim_rows) * 100
        min_mre = min(r["mre"] for r in sim_rows) * 100
        n_over = sum(1 for r in sim_rows if r["direction"] == "over")
        n_under = len(sim_rows) - n_over
        print(f"\n  L2 Accuracy ({len(sim_rows)} simulatable strategies):")
        print(f"    Avg MRE = {avg_mre:.1f}%  |  Min = {min_mre:.1f}%  |  Max = {max_mre:.1f}%")
        print(f"    Over-estimate: {n_over}  |  Under-estimate: {n_under}")

    # L2.5 / L3 summary
    for level, key in [("L2.5", "l25_mre"), ("L3", "l3_mre")]:
        level_rows = [r for r in all_rows if r.get(key) is not None]
        if level_rows:
            avg = sum(r[key] for r in level_rows) / len(level_rows) * 100
            print(f"  {level} Accuracy ({len(level_rows)} strategies): Avg MRE = {avg:.1f}%")

    # ── Error source analysis ──
    err_rows = [r for r in all_rows if r.get("err_total") is not None]
    if err_rows:
        print(f"\n  Error Source Analysis ({len(err_rows)} strategies):")
        print(f"  {'strategy':35s} {'total_err':>10s} {'err%':>7s} {'fixed_err':>10s} {'act_err':>10s} {'alloc_oh':>10s}")
        print("  " + "-" * 85)
        for r in err_rows:
            print(f"  {r['strategy']:35s} "
                  f"{format_bytes(abs(r['err_total'])):>10s} "
                  f"{r['err_pct']:>+6.1f}% "
                  f"{format_bytes(abs(r['err_fixed'])):>10s} "
                  f"{format_bytes(abs(r['err_act'])):>10s} "
                  f"{format_bytes(abs(r['err_alloc_overhead'])):>10s}")

    # ── CSV export ──
    csv_path = OUTPUT_DIR / "ex_sim_accuracy.csv"
    to_csv(all_rows, csv_path)
    print(f"\n  CSV -> {csv_path}")



if __name__ == "__main__":
    main()
