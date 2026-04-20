#!/usr/bin/env python
"""Example 1 — 静态显存估算 (L1 公式 + L2 图遍历)

核心卖点: 不需要实际训练，就能预测每个模型的 GPU 显存峰值和各分量组成。
输出:
  - 3 模型 × 显存分解表 (param / grad / optim / activation / peak)
  - L1 vs L2 对比
  - stacked bar chart (显存组成)
  - phase timeline chart (训练步骤各阶段显存变化)
"""

import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.capture import capture_graphs, count_fw_output_bytes, count_fw_outputs, graph_stats
from toolkit.models import ModelRegistry
from toolkit.output import print_comparison_table
from toolkit.simulation import estimate_from_config, estimate_training_peak
from toolkit.utils import format_bytes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
SEQ = 128
OUTPUT_DIR = Path(__file__).with_name("outputs")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()

    # ── L1: config-based estimation (no GPU forward needed) ──────────
    print("=" * 70)
    print("  Part A: L1 Config-based Estimation (公式推导)")
    print("=" * 70)

    l1_rows = []
    for name in registry.list_models():
        config = registry.get_config(name)
        l1 = estimate_from_config(config, BATCH, SEQ)
        l1_rows.append({
            "model": name,
            "param": format_bytes(l1["param_bytes"]),
            "grad": format_bytes(l1["grad_bytes"]),
            "optim": format_bytes(l1["optimizer_bytes"]),
            "activation": format_bytes(l1["activation_bytes"]),
            "fw_peak": format_bytes(l1["fw_peak"]),
            "bw_peak": format_bytes(l1["bw_peak"]),
            "opt_peak": format_bytes(l1["opt_peak"]),
            "true_peak": format_bytes(l1["true_peak"]),
            "peak_phase": l1["peak_phase"],
        })

    print_comparison_table(l1_rows, title="L1 Config Estimation — Memory Breakdown")

    # ── L2: graph-based estimation (需要 capture FW/BW 图) ───────────
    print("\n" + "=" * 70)
    print("  Part B: L2 Graph-based Estimation (图遍历仿真)")
    print("=" * 70)

    l2_rows = []
    timeline_items = []
    graph_info_rows = []

    for name in registry.list_models():
        config = registry.get_config(name)
        model = registry.create_model(name).to(DEVICE).train()
        input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

        fw_gm, bw_gm = capture_graphs(
            model, input_ids, lambda out: out.loss,
            model_kwargs={"labels": input_ids},
        )
        l2 = estimate_training_peak(fw_gm, bw_gm, model)
        fw_stats = graph_stats(fw_gm)

        l2_rows.append({
            "model": name,
            "param": format_bytes(l2["param_bytes"]),
            "grad": format_bytes(l2["grad_bytes"]),
            "optim": format_bytes(l2["optimizer_bytes"]),
            "fw_act_peak": format_bytes(l2["fw_graph_peak"]),
            "bw_act_peak": format_bytes(l2["bw_graph_peak"]),
            "saved_act": format_bytes(l2["saved_act_bytes"]),
            "fw_peak": format_bytes(l2["fw_peak"]),
            "bw_peak": format_bytes(l2["bw_peak"]),
            "true_peak": format_bytes(l2["true_peak"]),
            "peak_phase": l2["peak_phase"],
        })

        graph_info_rows.append({
            "model": name,
            "fw_nodes": fw_stats["n_total"],
            "fw_views": fw_stats["n_view"],
            "fw_alloc_nodes": fw_stats["n_alloc"],
            "fw_alloc_bytes": format_bytes(fw_stats["total_alloc_bytes"]),
            "saved_tensors": count_fw_outputs(fw_gm),
            "saved_bytes": format_bytes(count_fw_output_bytes(fw_gm)),
        })

        timeline_items.append({"name": name, **l2})

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print_comparison_table(l2_rows, title="L2 Graph Estimation — Memory Breakdown")
    print_comparison_table(graph_info_rows, title="L2 Graph Capture — FW Graph Structure")

    # ── L1 vs L2 对比 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Part C: L1 vs L2 Comparison")
    print("=" * 70)

    compare_rows = []
    for name in registry.list_models():
        config = registry.get_config(name)
        l1 = estimate_from_config(config, BATCH, SEQ)
        l2_item = next(t for t in timeline_items if t["name"] == name)
        diff_pct = (l1["true_peak"] - l2_item["true_peak"]) / l2_item["true_peak"] * 100
        compare_rows.append({
            "model": name,
            "L1_peak": format_bytes(l1["true_peak"]),
            "L2_peak": format_bytes(l2_item["true_peak"]),
            "L1_vs_L2": f"{diff_pct:+.1f}%",
            "L1_phase": l1["peak_phase"],
            "L2_phase": l2_item["peak_phase"],
        })

    print_comparison_table(compare_rows, title="L1 vs L2 — Peak Estimation Comparison")



if __name__ == "__main__":
    main()
