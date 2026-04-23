#!/usr/bin/env python
"""Horizontal simulation-method comparison.

Compares config formula, naive graph shape-sum, L2 live-range, L2.5
fusion-only, L2.5 safe-reuse, L3 Scheduler, and runtime profiling.

Presets:
  --quick   GPT-2 4L/256H,  B=4, S=128   (fast smoke)
  --medium  GPT-2 8L/512H,  B=8, S=256   (stronger F8/F9 draft)
  default   GPT-2/LLaMA/Mistral paper configs, B=8, S=512

Outputs:
  toolkit_examples/outputs/ex_horizontal_comparison.csv
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.capture import capture_graphs, capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.output import print_comparison_table, to_csv
from toolkit.profiler import measure_phased
from toolkit.simulation import (
    estimate_from_config,
    estimate_inductor_training_peak,
    estimate_shape_sum_peak,
    estimate_training_peak,
    make_level_stub,
)
from toolkit.strategy import (
    clear_memory_budget,
    get_partition_fn,
    set_memory_budget,
    wrap_with_checkpoint,
    wrap_with_sac,
)
from toolkit.utils import format_bytes, setup_experiment_env

setup_experiment_env()

DEVICE = "cuda"
OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3, fused=True)
FUSED_OPTIMIZER = True
OUTPUT_DIR = Path(__file__).with_name("outputs")

QUICK_MODELS = {
    "gpt2": dict(n_layer=4, n_embd=256, n_head=4, n_inner=1024, n_positions=512),
}
MEDIUM_MODELS = {
    "gpt2": dict(n_layer=8, n_embd=512, n_head=8, n_inner=2048, n_positions=1024),
}
PAPER_MODELS = {
    "gpt2": dict(n_embd=768, n_layer=12, n_head=12, n_inner=3072, n_positions=1024),
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

STRATEGIES = [
    dict(name="aot_eager+default", mode="aot_eager", partition="default"),
    dict(name="aot_eager+min_cut", mode="aot_eager", partition="min_cut"),
    dict(name="inductor(b=1.0)", mode="inductor", budget=1.0),
    dict(name="inductor(b=0.5)", mode="inductor", budget=0.5),
    dict(name="inductor(b=0.0)", mode="inductor", budget=0.0, has_recomputation=True),
    dict(name="ac+inductor", mode="inductor", wrapping="ac", has_recomputation=True),
    dict(name="sac_mm+inductor", mode="inductor", wrapping="sac", has_recomputation=True),
]


def _cleanup():
    torch._dynamo.reset()
    clear_memory_budget()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )

    torch._dynamo.reset()
    return torch.compile(model, backend=backend, dynamic=True)


def _compile_inductor(model, budget=None):
    if budget is not None:
        set_memory_budget(budget)
    torch._dynamo.reset()
    return torch.compile(model, backend="inductor", dynamic=True)


def _direction(est_peak, runtime_peak):
    if est_peak is None or runtime_peak is None:
        return None
    return "over" if est_peak > runtime_peak else "under"


def _mre(est_peak, runtime_peak):
    if est_peak is None or not runtime_peak:
        return None
    return abs(est_peak - runtime_peak) / runtime_peak


def _method_stub(est: dict, *, prefix: str | None = None) -> dict | None:
    if prefix is None:
        return est
    true_peak = est.get(f"{prefix}_true_peak")
    if true_peak is None:
        return None
    return {
        "true_peak": true_peak,
        "fw_peak": est.get(f"{prefix}_fw_peak", 0),
        "bw_peak": est.get(f"{prefix}_bw_peak", 0),
        "opt_peak": est.get(f"{prefix}_opt_peak", 0),
        "peak_phase": est.get(f"{prefix}_peak_phase"),
        "param_bytes": est["param_bytes"],
        "buffer_bytes": est.get("buffer_bytes", 0),
        "grad_bytes": est["grad_bytes"],
        "optimizer_bytes": est["optimizer_bytes"],
    }


def _add_method(row, name, est, runtime):
    peak = est.get("true_peak") if est else None
    row[f"{name}_true_peak"] = peak
    row[f"{name}_mre"] = _mre(peak, runtime.overall_peak)
    row[f"{name}_direction"] = _direction(peak, runtime.overall_peak)
    row[f"{name}_peak_phase"] = est.get("peak_phase") if est else None


def _apply_wrapping(model, registry, model_name, wrapping):
    if wrapping is None:
        return
    block_cls = registry.get_block_class_name(model_name)
    if wrapping == "ac":
        wrap_with_checkpoint(model, block_cls)
    elif wrapping == "sac":
        wrap_with_sac(model, block_cls, "save_matmuls")


def run_one(registry, model_name, overrides, strategy, batch, seq, repeats, warmup):
    config = registry.get_config(model_name, **overrides)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq), device=DEVICE)
    name = strategy["name"]
    label = f"{model_name}/{name}"
    print(f"  {label:42s} ...", end=" ", flush=True)

    # Runtime.
    model_rt = _new_model(registry, model_name, overrides)
    _apply_wrapping(model_rt, registry, model_name, strategy.get("wrapping"))
    if strategy["mode"] == "aot_eager":
        compiled = _compile_aot_eager(model_rt, get_partition_fn(strategy["partition"]))
    else:
        compiled = _compile_inductor(model_rt, strategy.get("budget"))
    optimizer = OPTIMIZER_CLS(model_rt.parameters(), **OPTIMIZER_KWARGS)
    runtime = measure_phased(
        label,
        lambda c=compiled, ids=input_ids: c(input_ids=ids, labels=ids).loss,
        optimizer,
        repeats=repeats,
        warmup=warmup,
        device=DEVICE,
    )
    del compiled, optimizer, model_rt
    _cleanup()

    # Simulation capture.
    model_sim = _new_model(registry, model_name, overrides)
    _apply_wrapping(model_sim, registry, model_name, strategy.get("wrapping"))

    row = {
        "model": model_name,
        "strategy": name,
        "batch": batch,
        "seq": seq,
        "rt_true_peak": runtime.overall_peak,
        "rt_fw_peak": runtime.fw_peak,
        "rt_bw_peak": runtime.bw_peak,
        "rt_opt_peak": runtime.opt_peak,
        "rt_peak_phase": runtime.peak_phase,
        "rt_step_ms": round(runtime.step_ms, 1),
    }

    l1 = estimate_from_config(
        config, batch, seq,
        optimizer=OPTIMIZER_CLS,
        fused_optimizer=FUSED_OPTIMIZER,
    )
    _add_method(row, "l1", l1, runtime)

    if strategy["mode"] == "aot_eager":
        fw_gm, bw_gm = capture_graphs(
            model_sim, input_ids, lambda out: out.loss,
            model_kwargs={"labels": input_ids},
            partition_fn=get_partition_fn(strategy["partition"]),
        )
        shape_sum = estimate_shape_sum_peak(
            fw_gm, bw_gm, model_sim,
            optimizer_cls=OPTIMIZER_CLS,
            fused_optimizer=FUSED_OPTIMIZER,
        )
        l2 = estimate_training_peak(
            fw_gm, bw_gm, model_sim,
            optimizer_cls=OPTIMIZER_CLS,
            fused_optimizer=FUSED_OPTIMIZER,
        )
        _add_method(row, "shape_sum", shape_sum, runtime)
        _add_method(row, "l2", l2, runtime)
        row["shape_sum_fw_bytes"] = shape_sum["shape_sum_fw_bytes"]
        row["shape_sum_bw_bytes"] = shape_sum["shape_sum_bw_bytes"]
    else:
        cap = capture_inductor_graphs(
            model_sim, input_ids, lambda out: out.loss,
            model_kwargs={"labels": input_ids},
            budget=strategy.get("budget"),
        )
        shape_sum = estimate_shape_sum_peak(
            cap["fw_gm"], cap["bw_gm"], model_sim,
            optimizer_cls=OPTIMIZER_CLS,
            fused_optimizer=FUSED_OPTIMIZER,
        )
        est = estimate_inductor_training_peak(
            cap, model_sim,
            optimizer_cls=OPTIMIZER_CLS,
            fused_optimizer=FUSED_OPTIMIZER,
            has_recomputation=strategy.get("has_recomputation"),
        )
        _add_method(row, "shape_sum", shape_sum, runtime)
        _add_method(row, "l2", est, runtime)
        _add_method(row, "l25_fusion", _method_stub(est, prefix="l25_fusion"), runtime)
        _add_method(row, "l25_safe", make_level_stub(est, "l25"), runtime)
        _add_method(row, "l3", make_level_stub(est, "l3"), runtime)
        row["shape_sum_fw_bytes"] = shape_sum["shape_sum_fw_bytes"]
        row["shape_sum_bw_bytes"] = shape_sum["shape_sum_bw_bytes"]
        row["l25_fw_reuses"] = est.get("l25_fw_reuses")
        row["l25_bw_reuses"] = est.get("l25_bw_reuses")
        row["has_recomputation"] = est.get("has_recomputation")

    print(
        f"RT={format_bytes(runtime.overall_peak):>9s} "
        f"L2={format_bytes(row['l2_true_peak']):>9s} "
        f"ShapeSum={format_bytes(row['shape_sum_true_peak']):>9s}"
    )
    del model_sim
    _cleanup()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use GPT-2 quick config")
    parser.add_argument("--medium", action="store_true", help="use stronger GPT-2 medium config")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    if args.quick and args.medium:
        raise SystemExit("--quick and --medium are mutually exclusive")

    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for runtime horizontal comparison")

    if args.quick:
        models = QUICK_MODELS
        default_batch, default_seq = 4, 128
        default_repeats, default_warmup = 1, 1
    elif args.medium:
        models = MEDIUM_MODELS
        default_batch, default_seq = 8, 256
        default_repeats, default_warmup = 1, 1
    else:
        models = PAPER_MODELS
        default_batch, default_seq = 8, 512
        default_repeats, default_warmup = 3, 2

    batch = args.batch if args.batch is not None else default_batch
    seq = args.seq if args.seq is not None else default_seq
    repeats = args.repeats if args.repeats is not None else default_repeats
    warmup = args.warmup if args.warmup is not None else default_warmup

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    rows = []

    print("=" * 80)
    print("  Horizontal Simulation Method Comparison")
    print(f"  Models={list(models)}  Batch={batch}  Seq={seq}")
    print("=" * 80)

    for model_name, overrides in models.items():
        print(f"\n-- {model_name} --")
        for strategy in STRATEGIES:
            try:
                rows.append(
                    run_one(
                        registry, model_name, overrides, strategy,
                        batch, seq, repeats, warmup,
                    )
                )
            except Exception as exc:
                print(f"FAILED ({exc})")
                rows.append({
                    "model": model_name,
                    "strategy": strategy["name"],
                    "batch": batch,
                    "seq": seq,
                    "error": str(exc),
                })
                _cleanup()

    csv_path = OUTPUT_DIR / "ex_horizontal_comparison.csv"
    to_csv(rows, csv_path)
    print(f"\nCSV -> {csv_path}")

    summary = []
    for r in rows:
        if r.get("error"):
            summary.append({"model": r["model"], "strategy": r["strategy"], "error": r["error"]})
            continue
        summary.append({
            "model": r["model"],
            "strategy": r["strategy"],
            "RT": format_bytes(r["rt_true_peak"]),
            "L1_MRE": f"{r['l1_mre'] * 100:.1f}%" if r.get("l1_mre") is not None else "N/A",
            "ShapeSum_MRE": f"{r['shape_sum_mre'] * 100:.1f}%" if r.get("shape_sum_mre") is not None else "N/A",
            "L2_MRE": f"{r['l2_mre'] * 100:.1f}%" if r.get("l2_mre") is not None else "N/A",
            "L2.5_MRE": f"{r['l25_safe_mre'] * 100:.1f}%" if r.get("l25_safe_mre") is not None else "N/A",
            "L3_MRE": f"{r['l3_mre'] * 100:.1f}%" if r.get("l3_mre") is not None else "N/A",
        })
    print_comparison_table(summary, title="Horizontal Method Comparison")


if __name__ == "__main__":
    main()
