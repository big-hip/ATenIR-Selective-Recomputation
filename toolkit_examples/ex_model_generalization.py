#!/usr/bin/env python
"""实验 3 — 多模型通用性验证

论文 §5.3: 验证 L2/L2.5/L3 仿真在 GPT-2 / LLaMA / Mistral 三种架构上的精度。

模型配置 (放大版, 适配 A6000 48GB):
  GPT-2:   12L/768H/12heads/3072I   — 原版 GPT-2 Small (124M params)
  LLaMA:   16L/2048H/16heads/8kv/5504I — ~870M params
  Mistral: 24L/1536H/12heads/4kv/4096I — ~550M params (deeper, narrower, GQA)

Batch: 8, Seq: 512
优化器: Adam(fused=True)

策略:
  eager, aot_eager+default, inductor(b=1.0), inductor(b=0.0),
  ac+inductor, sac_mm+inductor

输出:
  T1: 3模型×6策略 → Runtime / L2 / L2.5 / L3 / MRE
  T2: 平均 MRE per model per level
  CSV: outputs/ex_model_generalization.csv
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
from toolkit.profiler import measure_phased, validate
from toolkit.simulation import estimate_inductor_training_peak, estimate_training_peak, make_level_stub
from toolkit.strategy import (
    clear_memory_budget,
    get_partition_fn,
    set_memory_budget,
    wrap_with_checkpoint,
    wrap_with_sac,
)

DEVICE = "cuda"
BATCH = 8
SEQ = 512
OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3, fused=True)
FUSED_OPTIMIZER = True
OUTPUT_DIR = Path(__file__).with_name("outputs")

# ── Model configurations (放大版) ──
MODELS = {
    "gpt2": dict(
        n_embd=768, n_layer=12, n_head=12, n_inner=3072,
        n_positions=1024,
    ),
    "llama": dict(
        hidden_size=2048, num_hidden_layers=16, num_attention_heads=16,
        intermediate_size=5504, num_key_value_heads=8,
        max_position_embeddings=1024,
    ),
    "mistral": dict(
        hidden_size=1536, num_hidden_layers=24, num_attention_heads=12,
        intermediate_size=4096, num_key_value_heads=4,
        max_position_embeddings=1024,
    ),
}

# ── Strategy definitions ──
# (name, sim_mode, wrapping, partition_name, budget)
STRATEGIES = [
    ("eager",              None,        None,  None,      None),
    ("aot_eager+default",  "aot_eager", None,  "default", None),
    ("inductor(b=1.0)",    "inductor",  None,  None,      1.0),
    ("inductor(b=0.0)",    "inductor",  None,  None,      0.0),
    ("ac+inductor",        "inductor",  "ac",  None,      None),
    ("sac_mm+inductor",    "inductor",  "sac", None,      None),
]


def _new_model(registry, model_name, overrides):
    return registry.create_model(model_name, **overrides).to(DEVICE).train()


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
    return measure_phased(label, forward_fn, optimizer, repeats=3, warmup=2, device=DEVICE)


def _cleanup():
    torch._dynamo.reset()
    clear_memory_budget()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()

    print("=" * 80)
    print("  实验 3: 多模型通用性验证")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Batch: {BATCH} | Seq: {SEQ} | Optimizer: Adam(fused=True)")
    print("=" * 80)

    all_rows = []

    for model_name, overrides in MODELS.items():
        config = registry.get_config(model_name, **overrides)
        block_cls = registry.get_block_class_name(model_name)
        _tmp = _new_model(registry, model_name, overrides)
        param_bytes = count_unique_params(_tmp)
        del _tmp; gc.collect(); torch.cuda.empty_cache()
        input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

        print(f"\n  ── {model_name} (~{param_bytes / 1e6:.0f}M params, {format_bytes(param_bytes)}) ──")

        for strat_name, sim_mode, wrapping, partition_name, budget in STRATEGIES:
            label = f"{model_name}/{strat_name}"
            print(f"    {label:45s} ...", end=" ", flush=True)
            try:
                gc.collect()
                torch.cuda.empty_cache()

                # ── Runtime measurement ──
                m_rt = _new_model(registry, model_name, overrides)
                if wrapping == "ac":
                    wrap_with_checkpoint(m_rt, block_cls)
                elif wrapping == "sac":
                    wrap_with_sac(m_rt, block_cls, "save_matmuls")

                if sim_mode == "aot_eager":
                    compiled = _compile_aot_eager(m_rt, get_partition_fn(partition_name))
                elif sim_mode == "inductor":
                    compiled = _compile_inductor(m_rt, budget=budget)
                else:
                    compiled = m_rt

                opt = OPTIMIZER_CLS(m_rt.parameters(), **OPTIMIZER_KWARGS)
                rt = _measure(label, lambda c=compiled, ids=input_ids: c(input_ids=ids, labels=ids).loss, opt)

                del compiled, opt, m_rt
                _cleanup()

                # ── Simulation ──
                l2_peak = None
                l25_peak = None
                l3_peak = None
                val_l2 = None
                val_l25 = None
                val_l3 = None

                if sim_mode == "aot_eager":
                    m_sim = _new_model(registry, model_name, overrides)
                    if wrapping == "ac":
                        wrap_with_checkpoint(m_sim, block_cls)
                    elif wrapping == "sac":
                        wrap_with_sac(m_sim, block_cls, "save_matmuls")
                    fw_gm, bw_gm = capture_graphs(
                        m_sim, input_ids, lambda out: out.loss,
                        model_kwargs={"labels": input_ids},
                        partition_fn=get_partition_fn(partition_name),
                    )
                    est = estimate_training_peak(
                        fw_gm, bw_gm, m_sim,
                        optimizer_cls=OPTIMIZER_CLS,
                        fused_optimizer=FUSED_OPTIMIZER,
                    )
                    l2_peak = est["true_peak"]
                    val_l2 = validate(est, rt)
                    del m_sim, fw_gm, bw_gm
                    gc.collect()
                    torch.cuda.empty_cache()

                elif sim_mode == "inductor":
                    m_sim = _new_model(registry, model_name, overrides)
                    if wrapping == "ac":
                        wrap_with_checkpoint(m_sim, block_cls)
                    elif wrapping == "sac":
                        wrap_with_sac(m_sim, block_cls, "save_matmuls")
                    cap = capture_inductor_graphs(
                        m_sim, input_ids, lambda out: out.loss,
                        model_kwargs={"labels": input_ids},
                        budget=budget,
                    )
                    est = estimate_inductor_training_peak(
                        cap, m_sim,
                        optimizer_cls=OPTIMIZER_CLS,
                        fused_optimizer=FUSED_OPTIMIZER,
                        has_recomputation=(
                            True if (wrapping in ("ac", "sac")
                                     or (budget is not None and budget < 0.5))
                            else None
                        ),
                    )
                    l2_peak = est["true_peak"]
                    val_l2 = validate(est, rt)
                    l25_stub = make_level_stub(est, "l25")
                    if l25_stub is not None:
                        l25_peak = est["l25_true_peak"]
                        val_l25 = validate(l25_stub, rt)
                    l3_stub = make_level_stub(est, "l3")
                    if l3_stub is not None:
                        l3_peak = est["l3_true_peak"]
                        val_l3 = validate(l3_stub, rt)
                    del m_sim, cap
                    gc.collect()
                    torch.cuda.empty_cache()

                # ── Row ──
                row = {
                    "model": model_name,
                    "strategy": strat_name,
                    "rt_true_peak": rt.overall_peak,
                    "rt_fw_peak": rt.fw_peak,
                    "rt_bw_peak": rt.bw_peak,
                    "rt_opt_peak": rt.opt_peak,
                    "rt_peak_phase": rt.peak_phase,
                    "l2_true_peak": l2_peak,
                    "l25_true_peak": l25_peak,
                    "l3_true_peak": l3_peak,
                    "l2_mre": val_l2.mre_allocated if val_l2 else None,
                    "l25_mre": val_l25.mre_allocated if val_l25 else None,
                    "l3_mre": val_l3.mre_allocated if val_l3 else None,
                    "l2_direction": val_l2.direction if val_l2 else None,
                    "l25_direction": val_l25.direction if val_l25 else None,
                    "l3_direction": val_l3.direction if val_l3 else None,
                    "step_ms": round(rt.step_ms, 1),
                }
                all_rows.append(row)

                rt_str = format_bytes(rt.overall_peak)
                l2_str = format_bytes(l2_peak) if l2_peak else "N/A"
                l25_str = format_bytes(l25_peak) if l25_peak else "N/A"
                l3_str = format_bytes(l3_peak) if l3_peak else "N/A"
                l2_mre = f"{val_l2.mre_allocated*100:.1f}%" if val_l2 else "N/A"
                l25_mre = f"{val_l25.mre_allocated*100:.1f}%" if val_l25 else "N/A"
                l3_mre = f"{val_l3.mre_allocated*100:.1f}%" if val_l3 else "N/A"
                print(f"RT={rt_str:>9s}  L2={l2_str:>9s}  L2.5={l25_str:>9s}  L3={l3_str:>9s}  "
                      f"MRE_L2={l2_mre:>6s}  MRE_L2.5={l25_mre:>6s}  MRE_L3={l3_mre:>6s}")

            except Exception as e:
                print(f"FAILED ({e})")
                import traceback
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
                _cleanup()

    # ══════════════════════════════════════════════════════════════════
    #  Output
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)

    summary = []
    for r in all_rows:
        summary.append({
            "model": r["model"],
            "strategy": r["strategy"],
            "Runtime": format_bytes(r["rt_true_peak"]),
            "L2_peak": format_bytes(r["l2_true_peak"]) if r["l2_true_peak"] else "N/A",
            "L2.5_peak": format_bytes(r["l25_true_peak"]) if r.get("l25_true_peak") else "N/A",
            "L3_peak": format_bytes(r["l3_true_peak"]) if r["l3_true_peak"] else "N/A",
            "L2_MRE": f"{r['l2_mre']*100:.1f}%" if r["l2_mre"] is not None else "N/A",
            "L2.5_MRE": f"{r['l25_mre']*100:.1f}%" if r.get("l25_mre") is not None else "N/A",
            "L3_MRE": f"{r['l3_mre']*100:.1f}%" if r["l3_mre"] is not None else "N/A",
            "phase": r["rt_peak_phase"],
        })
    print_comparison_table(summary, title="3模型×6策略 — L2/L2.5/L3 vs Runtime")

    # T2: Average MRE per model per level
    print("\n  Average MRE per Model per Simulation Level")
    print(f"  {'Model':10s} {'L2':>10s} {'L2.5':>10s} {'L3':>10s}")
    print("  " + "-" * 35)
    for model_name in MODELS:
        model_rows = [r for r in all_rows if r["model"] == model_name]
        l2_mres = [r["l2_mre"] for r in model_rows if r["l2_mre"] is not None]
        l25_mres = [r["l25_mre"] for r in model_rows if r.get("l25_mre") is not None]
        l3_mres = [r["l3_mre"] for r in model_rows if r["l3_mre"] is not None]
        l2_avg = f"{sum(l2_mres)/len(l2_mres)*100:.1f}%" if l2_mres else "N/A"
        l25_avg = f"{sum(l25_mres)/len(l25_mres)*100:.1f}%" if l25_mres else "N/A"
        l3_avg = f"{sum(l3_mres)/len(l3_mres)*100:.1f}%" if l3_mres else "N/A"
        print(f"  {model_name:10s} {l2_avg:>10s} {l25_avg:>10s} {l3_avg:>10s}")

    csv_path = OUTPUT_DIR / "ex_model_generalization.csv"
    to_csv(all_rows, csv_path)
    print(f"\n  CSV -> {csv_path}")



if __name__ == "__main__":
    main()
