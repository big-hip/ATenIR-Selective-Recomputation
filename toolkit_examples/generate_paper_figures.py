#!/usr/bin/env python
"""Generate publication-quality figures from experiment CSV data.

Usage::

    python toolkit_examples/generate_paper_figures.py

Output directory: toolkit_examples/outputs/paper_figures/
Formats: PDF (vector) + PNG (300 dpi raster)

Prerequisites: run ex_sim_accuracy, ex_peak_phase, ex_model_generalization,
and optionally ex_horizontal_comparison first to populate the CSV/data files.
Missing CSVs are skipped gracefully.

Figure mapping:
  F1: Memory Composition Breakdown        — from L1 config estimation (no CSV)
  F2: 12-Strategy Overview + Pareto        — from ex_sim_accuracy.csv
  F3: Multi-Level Peak Comparison          — from ex_sim_accuracy.csv
  F4: Simulation Accuracy MRE              — from ex_sim_accuracy.csv
  F5: Peak Phase Heatmap                   — from ex_peak_phase.csv
  F6: Three-Phase Stacked Bar              — from ex_peak_phase.csv
  F7: Model Generalization Heatmap         — from ex_model_generalization.csv
  F8: Horizontal Method Comparison         — from ex_horizontal_comparison.csv
  F9: L2.5 Ablation                        — from ex_horizontal_comparison.csv
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.utils import setup_experiment_env
setup_experiment_env(enable_tf32=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolkit.output.pub_style import paper_style, savefig_pub, MB
from toolkit.output.pub_charts import (
    plot_f1_composition,
    plot_f2_strategy_overview,
    plot_f3_peak_comparison,
    plot_f4_mre,
    plot_f5_peak_phase_heatmap,
    plot_f5_merged,
    plot_f6_phase_stack,
    plot_f6_merged,
    plot_f7_model_heatmap,
    plot_f7_merged,
    plot_f8_horizontal_methods,
    plot_f9_l25_ablation,
)

DATA_DIR = Path(__file__).with_name("outputs")
OUT_DIR = DATA_DIR / "paper_figures"


# ── CSV loader ────────────────────────────────────────────────────
def _load_csv(name: str):
    """Load a CSV from DATA_DIR, return list of dicts. Returns [] if missing."""
    p = DATA_DIR / name
    if not p.exists():
        print(f"  [SKIP] {name} not found")
        return []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for k, v in row.items():
            if v == "" or v is None:
                row[k] = None
                continue
            try:
                row[k] = int(v)
            except (ValueError, TypeError):
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
    return rows


# ── F1: Memory Composition ───────────────────────────────────────
def _gen_f1():
    """F1: Memory Composition — L1 config estimation for 3 enlarged models.

    Uses the same model configs as ex_model_generalization to show
    param/grad/optim/activation breakdown for GPT-2 124M, LLaMA ~870M,
    Mistral ~870M.
    """
    try:
        from toolkit.models import ModelRegistry
        from toolkit.simulation import estimate_from_config
    except ImportError:
        print("  [SKIP] F1: cannot import toolkit")
        return

    registry = ModelRegistry()
    batch, seq = 8, 512

    # Use enlarged model configs matching ex_model_generalization
    model_configs = {
        "GPT-2 (124M)": dict(n_embd=768, n_layer=12, n_head=12, n_inner=3072,
                              n_positions=1024),
        "LLaMA (~870M)": dict(hidden_size=2048, num_hidden_layers=16,
                               num_attention_heads=16, intermediate_size=5504,
                               num_key_value_heads=8, max_position_embeddings=1024),
        "Mistral (~550M)": dict(hidden_size=1536, num_hidden_layers=24,
                                 num_attention_heads=12, intermediate_size=4096,
                                 num_key_value_heads=4, max_position_embeddings=1024),
    }
    model_names = {"GPT-2 (124M)": "gpt2", "LLaMA (~870M)": "llama",
                   "Mistral (~550M)": "mistral"}

    comp_rows = []
    for display_name, overrides in model_configs.items():
        base_name = model_names[display_name]
        config = registry.get_config(base_name, **overrides)
        est = estimate_from_config(config, batch, seq,
                                   optimizer="adam", fused_optimizer=True)
        comp_rows.append({
            "name": display_name,
            "param_bytes": est["param_bytes"],
            "grad_bytes": est["grad_bytes"],
            "optimizer_bytes": est["optimizer_bytes"],
            "true_peak": est["true_peak"],
        })

    if not comp_rows:
        print("  [SKIP] F1: no models")
        return

    fig = plot_f1_composition(comp_rows)
    savefig_pub(fig, OUT_DIR / "F1_composition")
    plt.close(fig)
    print("  [OK] F1_composition")


# Model subtitle shared by F2/F3/F4 (from ex_sim_accuracy experiment)
_SIM_SUBTITLE = "LLaMA ~870M  |  B=8, S=512  |  Adam (fused)"


# ── F2: Strategy Overview + Pareto ───────────────────────────────
def _gen_f2():
    """F2: 12-Strategy Overview bar chart + Memory-Time Pareto scatter."""
    rows = _load_csv("ex_sim_accuracy.csv")
    if not rows:
        return

    fig = plot_f2_strategy_overview(rows, subtitle=_SIM_SUBTITLE)
    savefig_pub(fig, OUT_DIR / "F2_strategy_overview")
    plt.close(fig)
    print("  [OK] F2_strategy_overview")


# ── F3: Multi-Level Peak Comparison ──────────────────────────────
def _gen_f3():
    """F3: Grouped bar of RT vs L2 vs L2.5 vs L3 for simulatable strategies."""
    rows = _load_csv("ex_sim_accuracy.csv")
    if not rows:
        return

    fig = plot_f3_peak_comparison(rows, subtitle=_SIM_SUBTITLE)
    savefig_pub(fig, OUT_DIR / "F3_peak_comparison")
    plt.close(fig)
    print("  [OK] F3_peak_comparison")


# ── F4: Simulation Accuracy MRE ──────────────────────────────────
def _gen_f4():
    """F4: MRE grouped bar for L2/L2.5/L3 per strategy."""
    rows = _load_csv("ex_sim_accuracy.csv")
    if not rows:
        return

    fig = plot_f4_mre(rows, subtitle=_SIM_SUBTITLE)
    savefig_pub(fig, OUT_DIR / "F4_mre")
    plt.close(fig)
    print("  [OK] F4_mre")


# ── F5: Peak Phase Heatmap ───────────────────────────────────────
def _gen_f5():
    """F5: Merged 2×2 heatmap of peak phase for all strategies."""
    rows = _load_csv("ex_peak_phase.csv")
    if not rows:
        return

    # Merged figure (primary)
    fig = plot_f5_merged(rows)
    savefig_pub(fig, OUT_DIR / "F5_peak_phase_merged")
    print("  [OK] F5_peak_phase_merged")


# ── F6: Three-Phase Stacked Bar ──────────────────────────────────
def _gen_f6():
    """F6: Merged stacked bar of fw/bw/opt for eager + inductor(b=1.0)."""
    rows = _load_csv("ex_peak_phase.csv")
    if not rows:
        return

    # Merged figure (primary) — 2 representative strategies per v3
    fig = plot_f6_merged(rows, strategies=["eager", "ac+inductor"],
                         subtitle="LLaMA ~870M  |  S=512")
    savefig_pub(fig, OUT_DIR / "F6_phase_stack_merged")
    print("  [OK] F6_phase_stack_merged")


# ── F7: Model Generalization Heatmap ─────────────────────────────
def _gen_f7():
    """F7: Merged 1×3 model × strategy MRE heatmap for L2/L2.5/L3."""
    rows = _load_csv("ex_model_generalization.csv")
    if not rows:
        return

    # Merged figure (primary)
    fig = plot_f7_merged(rows,
                         subtitle="B=8, S=512  |  Adam (fused)")
    savefig_pub(fig, OUT_DIR / "F7_model_heatmap_merged")
    print("  [OK] F7_model_heatmap_merged")


# ── F8: Horizontal Method Comparison ─────────────────────────────
def _gen_f8():
    """F8: Average MRE across L1/ShapeSum/L2/L2.5/L3 methods."""
    rows = _load_csv("ex_horizontal_comparison.csv")
    if not rows:
        return

    valid = [r for r in rows if not r.get("error")]
    if not valid:
        print("  [SKIP] F8: no successful horizontal rows")
        return

    fig = plot_f8_horizontal_methods(
        valid,
        subtitle="quick or paper config from ex_horizontal_comparison",
    )
    savefig_pub(fig, OUT_DIR / "F8_horizontal_methods")
    plt.close(fig)
    print("  [OK] F8_horizontal_methods")


# ── F9: L2.5 Ablation ────────────────────────────────────────────
def _gen_f9():
    """F9: L2 → fusion-only → safe-reuse → L3 ablation."""
    rows = _load_csv("ex_horizontal_comparison.csv")
    if not rows:
        return

    valid = [r for r in rows if not r.get("error")]
    if not valid:
        print("  [SKIP] F9: no successful horizontal rows")
        return

    fig = plot_f9_l25_ablation(
        valid,
        subtitle="quick or paper config from ex_horizontal_comparison",
    )
    savefig_pub(fig, OUT_DIR / "F9_l25_ablation")
    plt.close(fig)
    print("  [OK] F9_l25_ablation")


# ── Main ─────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old files to avoid stale standalone figures
    for old in OUT_DIR.glob("*"):
        if old.is_file():
            old.unlink()
    print("=" * 60)
    print("  Generating publication-quality figures")
    print(f"  Output: {OUT_DIR}/")
    print("=" * 60)

    with paper_style():
        _gen_f1()
        _gen_f2()
        _gen_f3()
        _gen_f4()
        _gen_f5()
        _gen_f6()
        _gen_f7()
        _gen_f8()
        _gen_f9()

    # List generated files
    files = sorted(OUT_DIR.glob("*.*"))
    print(f"\n  Generated {len(files)} files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s}  {size_kb:6.1f} KB")
    print()


if __name__ == "__main__":
    main()
