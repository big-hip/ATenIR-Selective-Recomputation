"""Publication-quality chart functions for the thesis (F1–F7).

Each function takes structured data and returns a matplotlib Figure.
All honour the ``paper_style()`` context.

Figure mapping:
  F1: Memory Composition Breakdown        — from L1 config estimation
  F2: 12-Strategy Overview + Pareto        — from ex_sim_accuracy.csv
  F3: Multi-Level Peak Comparison (RT/L2/L2.5/L3)  — from ex_sim_accuracy.csv
  F4: Simulation Accuracy MRE              — from ex_sim_accuracy.csv
  F5: Peak Phase Heatmap (batch×optimizer) — from ex_peak_phase.csv
  F6: Three-Phase Stacked Bar (fw/bw/opt)  — from ex_peak_phase.csv
  F7: Model Generalization Heatmap         — from ex_model_generalization.csv

Usage::

    from toolkit.output.pub_style import paper_style, savefig_pub
    from toolkit.output.pub_charts import plot_f1_composition

    with paper_style():
        fig = plot_f1_composition(rows)
        savefig_pub(fig, "outputs/F1_composition")
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

from .pub_style import (
    COLUMN_SINGLE, COLUMN_DOUBLE, COLUMN_THESIS,
    COLORS, LEVEL_COLORS, GROUP_COLORS, GROUP_LABELS,
    COMP_COLORS, LEVEL_HATCHES, MODEL_COLORS, ANNOT_SIZE,
    LEVEL_MARKERS,
    MB, short_strategy_name,
)

Row = Dict[str, Any]


GB = 1024 ** 3


def _to_float(v, default=0.0):
    """Safely convert to float."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _auto_unit(max_bytes: float):
    """Return (divisor, label) — GB if max > 1 GiB, else MB."""
    if max_bytes > GB:
        return GB, "GB"
    return MB, "MB"


# ═══════════════════════════════════════════════════════════════════
#  F1: Memory Composition Breakdown  (horizontal stacked bar)
# ═══════════════════════════════════════════════════════════════════
def plot_f1_composition(rows: List[Row], fig_width: float = COLUMN_THESIS) -> plt.Figure:
    """Horizontal stacked bar showing param/grad/optim/activation for each model.

    Expected row keys: name/model, param_bytes, grad_bytes, optimizer_bytes,
    true_peak (activation = true_peak - others).
    """
    labels = [r.get("name") or r.get("model", "?") for r in rows]
    n = len(labels)

    comps = []
    for r in rows:
        p = _to_float(r.get("param_bytes", 0))
        g = _to_float(r.get("grad_bytes", 0))
        o = _to_float(r.get("optimizer_bytes", 0))
        tp = _to_float(r.get("true_peak", 0))
        a = max(0, tp - p - g - o)
        comps.append((p, g, o, a))  # raw bytes; will divide later

    max_bytes = max((sum(c) for c in comps), default=0)
    unit_div, unit_label = _auto_unit(max_bytes)
    comps = [(p / unit_div, g / unit_div, o / unit_div, a / unit_div)
             for p, g, o, a in comps]

    comp_names = ["Parameters", "Gradients", "Optimizer", "Activation"]
    comp_cols = [COMP_COLORS["param"], COMP_COLORS["grad"],
                 COMP_COLORS["optimizer"], COMP_COLORS["activation"]]

    fig, ax = plt.subplots(figsize=(fig_width, 2.1), constrained_layout=True)
    y = np.arange(n)
    lefts = np.zeros(n)

    for idx, (cname, color) in enumerate(zip(comp_names, comp_cols)):
        vals = np.array([c[idx] for c in comps])
        ax.barh(y, vals, left=lefts, height=0.55,
                color=color, edgecolor="none",
                label=cname)
        for j, (v, l) in enumerate(zip(vals, lefts)):
            total = sum(comps[j])
            pct = v / total * 100 if total else 0
            if pct > 15:
                ax.text(l + v / 2, j, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=ANNOT_SIZE,
                        color="white")
        lefts += vals

    for j in range(n):
        total = sum(comps[j])
        ax.text(total + total * 0.015, j, f"{total:.1f} {unit_label}",
                ha="left", va="center", fontsize=ANNOT_SIZE)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Memory ({unit_label})")
    ax.set_title("Memory Composition Breakdown")
    ax.legend(loc="upper right", ncol=2)
    ax.invert_yaxis()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F2: 12-Strategy Overview + Pareto (2 subplots side by side)
# ═══════════════════════════════════════════════════════════════════
def _pareto_front(times, mems):
    """Return indices on the Pareto front (minimize both)."""
    pts = sorted(range(len(times)), key=lambda i: (times[i], mems[i]))
    front = []
    best_mem = float("inf")
    for i in pts:
        if mems[i] < best_mem:
            front.append(i)
            best_mem = mems[i]
    return front


def plot_f2_strategy_overview(rows: List[Row],
                              fig_width: float = COLUMN_THESIS,
                              subtitle: str = "") -> plt.Figure:
    """Two-panel figure: (a) grouped vertical bars, (b) Pareto scatter.

    (a) groups strategies by G1/G2/G3 with intra-group sorting and
    per-group baseline dashed lines, so different baselines are explicit.

    Expected keys: strategy, group, rt_true_peak, rt_step_ms.
    """
    # --- prepare data ---
    raw_peaks_all = [_to_float(r.get("rt_true_peak", 0)) for r in rows]
    unit_div, unit_label = _auto_unit(max(raw_peaks_all) if raw_peaks_all else 0)

    # split into groups, sort within each group by peak descending
    group_order = ["G1", "G2", "G3"]
    grouped = {g: [] for g in group_order}
    for r in rows:
        g = r.get("group", "G1")
        if g in grouped:
            grouped[g].append(r)
    for g in group_order:
        grouped[g].sort(key=lambda r: _to_float(r.get("rt_true_peak", 0)), reverse=True)

    # flatten with gaps between groups
    names, peaks, times, groups_flat, colors_bar = [], [], [], [], []
    x_positions = []
    pos = 0
    group_spans = {}  # g -> (start_pos, end_pos, baseline_peak)
    for gi, g in enumerate(group_order):
        if not grouped[g]:
            continue
        if pos > 0:
            pos += 0.6  # gap between groups
        start = pos
        baseline_peak = None
        for r in grouped[g]:
            nm = short_strategy_name(r.get("strategy", ""))
            pk = _to_float(r.get("rt_true_peak", 0)) / unit_div
            tm = _to_float(r.get("rt_step_ms", 0))
            if baseline_peak is None:
                baseline_peak = pk  # first (highest) in group = baseline
            names.append(nm)
            peaks.append(pk)
            times.append(tm)
            groups_flat.append(g)
            colors_bar.append(GROUP_COLORS.get(g, COLORS["gray"]))
            x_positions.append(pos)
            pos += 1
        group_spans[g] = (start, pos - 1, baseline_peak)

    n = len(names)
    x = np.array(x_positions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 3.4),
                                    gridspec_kw={"width_ratios": [1.5, 1]},
                                    constrained_layout=True)

    # (a) Grouped vertical bar chart
    ax1.bar(x, peaks, color=colors_bar, edgecolor="none", width=0.72)
    ax1.grid(axis="y", alpha=0.25, lw=0.4)

    # per-group baseline dashed line + light background
    for g, (s, e, bl) in group_spans.items():
        c = GROUP_COLORS.get(g, "#999")
        ax1.axhline(bl, xmin=0, xmax=1, color=c, ls="--", lw=0.7, alpha=0.3)
        ax1.axvspan(s - 0.45, e + 0.45, alpha=0.04, color=c, zorder=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    ax1.set_ylabel(f"Peak Memory ({unit_label})")
    ax1.set_title("(a) Strategy Comparison")
    ax1.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g])
               for g in group_order if g in set(groups_flat)]
    ax1.legend(handles=handles, loc="upper right", fontsize=6)

    # (b) Pareto scatter — smart label placement
    front = _pareto_front(times, peaks)
    front_set = set(front)
    # pre-sort points for smarter offset alternation
    labeled_pts = []
    for i, (t, m, g, nm) in enumerate(zip(times, peaks, groups_flat, names)):
        c = GROUP_COLORS.get(g, COLORS["gray"])
        ax2.scatter(t, m, c=c, s=64, alpha=0.85,
                    edgecolors="white", linewidths=0.5, zorder=3)
        if i in front_set:
            labeled_pts.append((i, t, m, nm))
    # alternate label offsets to reduce overlap
    _offsets = [(6, -10), (-6, 8), (6, 6), (-6, -10)]
    for li, (i, t, m, nm) in enumerate(labeled_pts):
        ox, oy = _offsets[li % len(_offsets)]
        ax2.annotate(nm, (t, m), textcoords="offset points",
                     xytext=(ox, oy), fontsize=ANNOT_SIZE,
                     color="#333",
                     arrowprops=dict(arrowstyle="-", color="#BBB",
                                     lw=0.5, shrinkB=3))
    if len(front) >= 2:
        ft = sorted([(times[i], peaks[i]) for i in front])
        ax2.plot([p[0] for p in ft], [p[1] for p in ft],
                 color="#999", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax2.set_xlabel("Step Time (ms)")
    ax2.set_ylabel(f"Peak Memory ({unit_label})")
    ax2.set_title("(b) Memory-Time Tradeoff")
    # add margin so edge labels aren't clipped
    ax2.margins(0.08)

    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=ANNOT_SIZE,
                 fontstyle="italic", color="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F3: Parity Scatter — Simulated vs Runtime Peak Memory
# ═══════════════════════════════════════════════════════════════════
def plot_f3_peak_comparison(rows: List[Row],
                            fig_width: float = 4.1,
                            subtitle: str = "") -> plt.Figure:
    """Parity scatter plot: Simulated peak (y) vs Runtime peak (x).

    Each point is one (strategy, level) pair.  The y=x diagonal shows
    perfect prediction; shaded ±10 % bands show the acceptability zone.
    A statistics text-box reports MAPE per level.

    Expected keys: strategy, rt_true_peak, l2_true_peak,
    l25_true_peak (optional), l3_true_peak (optional).
    """
    sim_rows = [r for r in rows if _to_float(r.get("l2_true_peak")) > 0]
    if not sim_rows:
        sim_rows = rows

    levels = [
        ("L2",   "l2_true_peak"),
        ("L2.5", "l25_true_peak"),
        ("L3",   "l3_true_peak"),
    ]
    active = [(lbl, k) for lbl, k in levels
              if any(_to_float(r.get(k)) > 0 for r in sim_rows)]
    if not active:
        active = levels[:1]

    # collect all values for auto-unit
    all_vals = []
    for r in sim_rows:
        rt = _to_float(r.get("rt_true_peak", 0))
        for _, k in active:
            sv = _to_float(r.get(k, 0))
            if sv > 0:
                all_vals.extend([rt, sv])
    unit_div, unit_label = _auto_unit(max(all_vals) if all_vals else 0)

    fig, ax = plt.subplots(figsize=(fig_width, 3.8), constrained_layout=True)

    stat_lines = []
    for lbl, key in active:
        c = LEVEL_COLORS.get(lbl, COLORS["gray"])
        m = LEVEL_MARKERS.get(lbl, "o")
        rts, sims = [], []
        for r in sim_rows:
            rt = _to_float(r.get("rt_true_peak", 0))
            sv = _to_float(r.get(key, 0))
            if sv > 0 and rt > 0:
                rts.append(rt / unit_div)
                sims.append(sv / unit_div)
        if rts:
            ax.scatter(rts, sims, c=c, marker=m, s=72, alpha=0.75,
                       edgecolors="white", linewidths=0.6,
                       label=lbl, zorder=3)
            # compute MAPE for stats box
            errs = [abs(s - r) / r * 100 for s, r in zip(sims, rts) if r > 0]
            mape = sum(errs) / len(errs) if errs else 0
            stat_lines.append(f"{lbl}: MAPE = {mape:.1f}%")

    # y=x line via transAxes (always spans full axis range)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,
            color="#333", ls="-", lw=0.8, zorder=1)
    # ±10% shaded band
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    margin = (hi - lo) * 0.05
    lo -= margin
    hi += margin
    diag = np.array([lo, hi])
    ax.fill_between(diag, diag * 0.9, diag * 1.1,
                    color="#E0E0E0", alpha=0.4, zorder=0,
                    label="\u00b110%")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # statistics text-box
    if stat_lines:
        ax.text(0.05, 0.95, "\n".join(stat_lines),
                transform=ax.transAxes, fontsize=ANNOT_SIZE,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CCCCCC", lw=0.5, alpha=0.85))

    ax.set_xlabel(f"Runtime Peak ({unit_label})")
    ax.set_ylabel(f"Simulated Peak ({unit_label})")
    ax.set_title("Simulation vs. Runtime")
    ax.legend(loc="lower right")

    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=ANNOT_SIZE,
                 fontstyle="italic", color="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F4: Simulation Accuracy — MRE  (grouped bar L2/L2.5/L3)
# ═══════════════════════════════════════════════════════════════════
def plot_f4_mre(rows: List[Row],
                fig_width: float = COLUMN_THESIS,
                subtitle: str = "") -> plt.Figure:
    """Cleveland dot plot of MRE (%) for L2, L2.5, L3 per strategy.

    Sorted by average MRE ascending (best at top).  10% threshold shown
    as a subtle vertical band.  Much cleaner than grouped bars.

    Expected keys: strategy, mre, l25_mre, l3_mre,
    direction, l25_direction, l3_direction (optional).
    """
    sim_rows = [r for r in rows if _to_float(r.get("mre")) > 0]
    if not sim_rows:
        sim_rows = rows

    mre_specs = [
        ("L2",   "mre",     "direction"),
        ("L2.5", "l25_mre", "l25_direction"),
        ("L3",   "l3_mre",  "l3_direction"),
    ]
    active = [(lbl, k, dk) for lbl, k, dk in mre_specs
              if any(_to_float(r.get(k)) > 0 for r in sim_rows)]

    # sort by average MRE ascending (best at top)
    def _avg_mre(r):
        vals = [_to_float(r.get(k, 0)) * 100 for _, k, _ in active]
        nonzero = [v for v in vals if v > 0]
        return sum(nonzero) / len(nonzero) if nonzero else 0
    sim_rows = sorted(sim_rows, key=_avg_mre)

    names = [short_strategy_name(r.get("strategy", "")) for r in sim_rows]
    n = len(names)
    _markers = ["o", "s", "D"]  # circle, square, diamond

    fig, ax = plt.subplots(figsize=(fig_width, 3.0), constrained_layout=True)
    y = np.arange(n)

    # 10% threshold band (prominent but subtle)
    ax.axvspan(10, ax.get_xlim()[1] if ax.get_xlim()[1] > 10 else 30,
               alpha=0.06, color="#E74C3C", zorder=0)
    ax.axvline(10, color="#E74C3C", ls="--", lw=1.0, alpha=0.5, zorder=1)

    # alternating row background for readability
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color="#F5F5F5", zorder=0)

    for li, (lbl, key, dir_key) in enumerate(active):
        vals = [_to_float(r.get(key, 0)) * 100 for r in sim_rows]
        c = LEVEL_COLORS.get(lbl, COLORS["gray"])
        nonzero = [v for v in vals if v > 0]
        avg = sum(nonzero) / len(nonzero) if nonzero else 0
        lbl_text = f"{lbl}  (avg {avg:.1f}%)" if avg > 0 else lbl
        # connecting thin line from 0 to dot
        for i, v in enumerate(vals):
            if v > 0:
                ax.plot([0, v], [i, i], color=c, lw=0.6, alpha=0.3, zorder=1)
        # dot markers
        plot_vals = [v if v > 0 else np.nan for v in vals]
        ax.scatter(plot_vals, y, c=c, s=48, marker=_markers[li % 3],
                   edgecolors="white", linewidths=0.4, zorder=4,
                   label=lbl_text)

    ax.grid(axis="x", alpha=0.2, lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MRE (%)")
    ax.set_title("Simulation Accuracy — MRE")
    ax.set_xlim(left=0)
    # add text annotation for threshold
    ax.text(10.3, n - 0.5, "10%", fontsize=7, color="#E74C3C", alpha=0.7, va="top")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=ANNOT_SIZE,
                 fontstyle="italic", color="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F5: Peak Phase Heatmap (batch × optimizer → peak_phase)
# ═══════════════════════════════════════════════════════════════════
PHASE_COLORS = {"FW": COLORS["blue"], "BW": "#D55E00", "OPT": COLORS["purple"]}
PHASE_CODE = {"FW": 0, "BW": 1, "OPT": 2}
_OOM_COLOR = "#BDBDBD"


def plot_f5_peak_phase_heatmap(rows: List[Row],
                               strategy: str = "eager",
                               fig_width: float = COLUMN_DOUBLE) -> plt.Figure:
    """Heatmap: batch (y) × optimizer (x) → peak phase, for a given strategy.

    Expected keys: optimizer, batch, strategy, peak_phase, true_peak.
    OOM rows (peak_phase=="OOM") are shown as gray cells with "OOM" label.
    """
    # include ALL rows for this strategy (OOM + valid) to build grid
    strat_rows = [r for r in rows if r.get("strategy") == strategy]
    if not strat_rows:
        strat_rows = rows

    optimizers = list(OrderedDict.fromkeys(r.get("optimizer", "") for r in strat_rows))
    batches = sorted(set(int(_to_float(r.get("batch", 0))) for r in strat_rows))

    n_opt = len(optimizers)
    n_batch = len(batches)
    if n_opt == 0 or n_batch == 0:
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        return fig

    # build 2D arrays  (OOM → code 3)
    OOM_CODE = 3
    phase_arr = np.full((n_batch, n_opt), np.nan)
    peak_arr = np.full((n_batch, n_opt), np.nan)
    phase_labels = [[""]*n_opt for _ in range(n_batch)]

    for r in strat_rows:
        opt_name = r.get("optimizer", "")
        b_val = int(_to_float(r.get("batch", 0)))
        oi = optimizers.index(opt_name) if opt_name in optimizers else -1
        bi = batches.index(b_val) if b_val in batches else -1
        if oi >= 0 and bi >= 0:
            ph = r.get("peak_phase", "")
            if ph == "OOM" or r.get("true_peak") in (None, "", "None"):
                phase_arr[bi, oi] = OOM_CODE
                phase_labels[bi][oi] = "OOM"
            else:
                phase_arr[bi, oi] = PHASE_CODE.get(ph, -1)
                peak_arr[bi, oi] = _to_float(r.get("true_peak", 0))
                phase_labels[bi][oi] = ph

    # custom colormap: FW=blue, BW=vermillion, OPT=pink, OOM=gray
    cmap = mcolors.ListedColormap([PHASE_COLORS["FW"], PHASE_COLORS["BW"],
                                   PHASE_COLORS["OPT"], _OOM_COLOR])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(fig_width * 0.6, 0.5 * n_batch + 1.0))

    ax.imshow(phase_arr, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_opt))
    ax.set_yticks(np.arange(n_batch))
    ax.set_xticklabels(optimizers)
    ax.set_yticklabels([str(b) for b in batches])
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Batch Size")

    # cell annotations: phase label only (clean)
    for i in range(n_batch):
        for j in range(n_opt):
            ph = phase_labels[i][j]
            if ph:
                ax.text(j, i, ph, ha="center", va="center",
                        fontsize=ANNOT_SIZE, color="white")

    handles = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in ["FW", "BW", "OPT"]]
    handles.append(Patch(facecolor=_OOM_COLOR, label="OOM"))
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_title(f"Peak Phase — {strategy}")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F5 merged: 2×2 Peak Phase Heatmap for multiple strategies
# ═══════════════════════════════════════════════════════════════════
def plot_f5_merged(rows: List[Row],
                   strategies: Optional[List[str]] = None,
                   fig_width: float = COLUMN_THESIS) -> plt.Figure:
    """1×4 heatmap row: batch (y) × optimizer (x) → peak phase, one subplot per strategy.

    Args:
        rows: all rows from ex_peak_phase.csv
        strategies: list of strategy names (default: auto-detect up to 4)
    """
    if strategies is None:
        strategies = list(OrderedDict.fromkeys(r.get("strategy", "") for r in rows))
    strategies = strategies[:4]
    n = len(strategies)

    OOM_CODE = 3
    cmap = mcolors.ListedColormap([PHASE_COLORS["FW"], PHASE_COLORS["BW"],
                                   PHASE_COLORS["OPT"], _OOM_COLOR])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # short optimizer labels to avoid overlap
    _opt_short = {"Adam(fused)": "Fused", "Adam": "Adam", "SGD": "SGD"}

    fig, axes = plt.subplots(1, n, figsize=(fig_width, 2.8),
                              squeeze=False, constrained_layout=True)

    for idx, strat in enumerate(strategies):
        ax = axes[0][idx]
        strat_rows = [r for r in rows if r.get("strategy") == strat]
        if not strat_rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8,
                    transform=ax.transAxes)
            ax.set_title(strat, fontsize=7)
            continue

        optimizers = list(OrderedDict.fromkeys(r.get("optimizer", "") for r in strat_rows))
        batches = sorted(set(int(_to_float(r.get("batch", 0))) for r in strat_rows))
        n_opt, n_batch = len(optimizers), len(batches)

        phase_arr = np.full((n_batch, n_opt), np.nan)
        phase_labels = [[""] * n_opt for _ in range(n_batch)]

        for r in strat_rows:
            opt_name = r.get("optimizer", "")
            b_val = int(_to_float(r.get("batch", 0)))
            oi = optimizers.index(opt_name) if opt_name in optimizers else -1
            bi = batches.index(b_val) if b_val in batches else -1
            if oi >= 0 and bi >= 0:
                ph = r.get("peak_phase", "")
                if ph == "OOM" or r.get("true_peak") in (None, "", "None"):
                    phase_arr[bi, oi] = OOM_CODE
                    phase_labels[bi][oi] = "OOM"
                else:
                    phase_arr[bi, oi] = PHASE_CODE.get(ph, -1)
                    phase_labels[bi][oi] = ph

        ax.imshow(phase_arr, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(n_opt))
        ax.set_yticks(np.arange(n_batch))
        opt_labels = [_opt_short.get(o, o) for o in optimizers]
        ax.set_xticklabels(opt_labels, fontsize=7)
        ax.set_yticklabels([str(b) for b in batches])

        if idx == 0:
            ax.set_ylabel("Batch Size")
        else:
            ax.set_yticklabels([])

        # cell annotations: phase label only
        for i in range(n_batch):
            for j in range(n_opt):
                ph = phase_labels[i][j]
                if ph:
                    ax.text(j, i, ph, ha="center", va="center",
                            fontsize=ANNOT_SIZE, color="white")

        label = chr(97 + idx)  # a, b, c, d
        ax.set_title(f"({label}) {short_strategy_name(strat)}", pad=4)

    # shared legend — horizontal strip at top of figure, no cell occlusion
    handles = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in ["FW", "BW", "OPT"]]
    handles.append(Patch(facecolor=_OOM_COLOR, label="OOM"))
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06), fontsize=7,
               handlelength=1.0, handletextpad=0.3, columnspacing=1.2,
               frameon=False)
    # shared x-label
    fig.supxlabel("Optimizer", fontsize=9)
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F6: Three-Phase Stacked Bar (fw/bw/opt vs batch for each optimizer)
# ═══════════════════════════════════════════════════════════════════
def plot_f6_phase_stack(rows: List[Row],
                        strategy: str = "eager",
                        fig_width: float = COLUMN_DOUBLE) -> plt.Figure:
    """Grouped stacked bar: for each optimizer, bars per batch showing fw/bw/opt.

    Expected keys: optimizer, batch, strategy, fw_peak, bw_peak, opt_peak.
    """
    valid = [r for r in rows if r.get("peak_phase") not in ("OOM", None) and r.get("true_peak") is not None]
    strat_rows = [r for r in valid if r.get("strategy") == strategy]
    if not strat_rows:
        strat_rows = valid

    optimizers = list(OrderedDict.fromkeys(r.get("optimizer", "") for r in strat_rows))
    batches = sorted(set(int(_to_float(r.get("batch", 0))) for r in strat_rows))
    n_opt = len(optimizers)
    n_batch = len(batches)

    # auto unit
    max_val = max((_to_float(r.get("true_peak", 0)) for r in strat_rows), default=0)
    unit_div, unit_label = _auto_unit(max_val)

    fig, axes = plt.subplots(1, n_opt, figsize=(fig_width, 3.0), sharey=True)
    if n_opt == 1:
        axes = [axes]

    phase_names = ["fw_peak", "bw_peak", "opt_peak"]
    phase_labels = ["Forward", "Backward", "Optimizer"]
    phase_colors = [PHASE_COLORS["FW"], PHASE_COLORS["BW"], PHASE_COLORS["OPT"]]

    for oi, (opt_name, ax) in enumerate(zip(optimizers, axes)):
        x = np.arange(n_batch)
        bottoms = np.zeros(n_batch)

        for pi, (pk, plbl, pcol) in enumerate(zip(phase_names, phase_labels, phase_colors)):
            vals = []
            for b in batches:
                match = [r for r in strat_rows
                         if r.get("optimizer") == opt_name
                         and int(_to_float(r.get("batch", 0))) == b]
                vals.append(_to_float(match[0].get(pk, 0)) / unit_div if match else 0)
            vals_arr = np.array(vals)
            ax.bar(x, vals_arr, bottom=bottoms, width=0.6,
                   color=pcol, edgecolor="none",
                   label=plbl if oi == 0 else "")
            bottoms += vals_arr

        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches])
        ax.set_xlabel("Batch Size")
        ax.set_title(opt_name)

    axes[0].set_ylabel(f"Peak Memory ({unit_label})")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F6 merged: Two-strategy stacked bar (top / bottom rows)
# ═══════════════════════════════════════════════════════════════════
def plot_f6_merged(rows: List[Row],
                   strategies: Optional[List[str]] = None,
                   fig_width: float = COLUMN_THESIS,
                   subtitle: str = "") -> plt.Figure:
    """2×3 small-multiples stacked bar: rows=strategies, cols=optimizers.

    Each subplot: x=batch size, stacked FW/BW/OPT.  No hatching needed —
    optimizer identity comes from column position.  Very clean layout.

    Args:
        rows: all rows from ex_peak_phase.csv
        strategies: list of 2 strategy names (default: ["eager", "ac+inductor"])
    """
    if strategies is None:
        strategies = ["eager", "ac+inductor"]
    strategies = strategies[:2]

    valid = [r for r in rows
             if r.get("peak_phase") not in ("OOM", None) and r.get("true_peak") is not None]

    # detect optimizers from data
    all_optimizers = list(OrderedDict.fromkeys(r.get("optimizer", "") for r in valid
                                                if r.get("strategy") in strategies))
    n_strats = len(strategies)
    n_opt = len(all_optimizers)
    if n_strats == 0 or n_opt == 0:
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # auto unit
    max_val = max((_to_float(r.get("true_peak", 0)) for r in valid
                   if r.get("strategy") in strategies), default=0)
    unit_div, unit_label = _auto_unit(max_val)

    phase_names = ["fw_peak", "bw_peak", "opt_peak"]
    phase_labels = ["FW", "BW", "OPT"]
    phase_colors = [PHASE_COLORS["FW"], PHASE_COLORS["BW"], PHASE_COLORS["OPT"]]

    fig, axes = plt.subplots(n_strats, n_opt, figsize=(fig_width, 4.0),
                              sharey=True, sharex=True, constrained_layout=True)
    if n_strats == 1:
        axes = axes[np.newaxis, :]
    if n_opt == 1:
        axes = axes[:, np.newaxis]

    # collect totals per (strategy, optimizer, batch) for savings annotation
    totals = {}  # (si, oi, bi) -> total_peak
    for si, strat in enumerate(strategies):
        strat_rows = [r for r in valid if r.get("strategy") == strat]
        batches = sorted(set(int(_to_float(r.get("batch", 0))) for r in strat_rows))
        n_batch = len(batches)
        x = np.arange(n_batch)

        for oi, opt_name in enumerate(all_optimizers):
            ax = axes[si, oi]
            bottoms = np.zeros(n_batch)

            for pi, (pk, plbl, pcol) in enumerate(zip(phase_names, phase_labels, phase_colors)):
                vals = []
                for b in batches:
                    match = [r for r in strat_rows
                             if r.get("optimizer") == opt_name
                             and int(_to_float(r.get("batch", 0))) == b]
                    vals.append(_to_float(match[0].get(pk, 0)) / unit_div if match else 0)
                vals_arr = np.array(vals)
                label = plbl if si == 0 and oi == 0 else ""
                ax.bar(x, vals_arr, 0.65, bottom=bottoms, color=pcol,
                       edgecolor="white", linewidth=0.3, label=label)
                bottoms += vals_arr

            # store totals
            for bi in range(n_batch):
                totals[(si, oi, bi)] = bottoms[bi]

            ax.grid(axis="y", alpha=0.2, lw=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in batches], fontsize=7)

            # column titles (top row only) = optimizer name
            if si == 0:
                ax.set_title(opt_name, fontsize=8, pad=3)
            # row labels (left column only) = strategy name
            if oi == 0:
                ax.set_ylabel(f"{short_strategy_name(strat)}\n({unit_label})",
                              fontsize=7.5)

    # annotate savings on non-baseline rows (si > 0)
    baseline_si = 0
    for si in range(1, n_strats):
        strat_rows = [r for r in valid if r.get("strategy") == strategies[si]]
        batches = sorted(set(int(_to_float(r.get("batch", 0))) for r in strat_rows))
        n_batch = len(batches)
        for oi in range(n_opt):
            ax = axes[si, oi]
            for bi in range(n_batch):
                base_v = totals.get((baseline_si, oi, bi), 0)
                cur_v = totals.get((si, oi, bi), 0)
                if base_v > 0:
                    pct = (cur_v - base_v) / base_v * 100
                    sign = "" if pct >= 0 else ""
                    txt = f"{sign}{pct:+.0f}%"
                    ax.text(bi, cur_v + 0.5, txt, ha="center", va="bottom",
                            fontsize=6, color="#E74C3C" if pct < 0 else "#333",
                            fontweight="bold")

    # shared phase legend at top
    phase_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                     for c, l in zip(phase_colors, phase_labels)]
    fig.legend(handles=phase_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), fontsize=7,
               handlelength=1.0, handletextpad=0.3, columnspacing=1.5,
               frameon=False)
    fig.supxlabel("Batch Size", fontsize=9)
    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=ANNOT_SIZE,
                 fontstyle="italic", color="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F7: Model Generalization Heatmap (model × strategy → MRE)
# ═══════════════════════════════════════════════════════════════════
def plot_f7_model_heatmap(rows: List[Row],
                          level: str = "l2",
                          fig_width: float = COLUMN_DOUBLE) -> plt.Figure:
    """Heatmap of MRE (%) for model × strategy, with multi-level support.

    Expected keys: model, strategy, l2_mre, l25_mre, l3_mre.
    Args:
        level: "l2", "l25", or "l3" — which MRE level to display.
    """
    mre_key = f"{level}_mre"
    dir_key = f"{level}_direction"

    models = list(OrderedDict.fromkeys(r.get("model", "") for r in rows))
    strategies = list(OrderedDict.fromkeys(r.get("strategy", "") for r in rows))

    # Filter to strategies that have at least one non-None MRE
    valid_strategies = []
    for s in strategies:
        if any(_to_float(r.get(mre_key)) > 0
               for r in rows if r.get("strategy") == s):
            valid_strategies.append(s)
    if not valid_strategies:
        valid_strategies = strategies

    n_models = len(models)
    n_strats = len(valid_strategies)

    arr = np.zeros((n_models, n_strats))
    dir_arr = [[""] * n_strats for _ in range(n_models)]

    for r in rows:
        m = r.get("model", "")
        s = r.get("strategy", "")
        if m in models and s in valid_strategies:
            mi = models.index(m)
            si = valid_strategies.index(s)
            arr[mi, si] = _to_float(r.get(mre_key, 0)) * 100
            dir_arr[mi][si] = r.get(dir_key, "") or ""

    fig, ax = plt.subplots(
        figsize=(min(fig_width, 0.8 * n_strats + 1.5),
                 0.5 * n_models + 1.0))

    vmax = max(arr.max(), 15) if arr.size > 0 else 15
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(n_strats))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([short_strategy_name(s) for s in valid_strategies],
                       rotation=35, ha="right")
    ax.set_yticklabels(models)

    for i in range(n_models):
        for j in range(n_strats):
            v = arr[i, j]
            if v > 0:
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color="#999")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cb.set_label("MRE (%)")

    level_display = {"l2": "L2", "l25": "L2.5", "l3": "L3"}.get(level, level)
    ax.set_title(f"Model Generalization — {level_display} MRE")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F7 merged: 1×N grouped bar chart — MRE per strategy, grouped by model
# ═══════════════════════════════════════════════════════════════════
def plot_f7_merged(rows: List[Row],
                   levels: Optional[List[str]] = None,
                   fig_width: float = 4.8,
                   subtitle: str = "") -> plt.Figure:
    """N×1 vertical heatmap subplots: models (rows) × strategies (cols) → MRE (%).

    3×1 vertical layout preserves all 3 simulation levels for evidence chain.

    Args:
        rows: all rows from ex_model_generalization.csv
        levels: list of level prefixes, e.g. ["l2", "l25", "l3"]
    """
    if levels is None:
        levels = ["l2", "l25", "l3"]

    level_display = {"l2": "L2", "l25": "L2.5", "l3": "L3"}

    active_levels = [lv for lv in levels
                     if any(_to_float(r.get(f"{lv}_mre")) > 0 for r in rows)]
    if not active_levels:
        active_levels = levels[:1]
    n_lv = len(active_levels)

    models = list(OrderedDict.fromkeys(r.get("model", "") for r in rows))
    all_strategies = list(OrderedDict.fromkeys(r.get("strategy", "") for r in rows))

    valid_strategies = []
    for s in all_strategies:
        for lv in active_levels:
            if any(_to_float(r.get(f"{lv}_mre")) > 0
                   for r in rows if r.get("strategy") == s):
                valid_strategies.append(s)
                break
    if not valid_strategies:
        valid_strategies = all_strategies

    n_models = len(models)
    n_strats = len(valid_strategies)
    strat_names = [short_strategy_name(s) for s in valid_strategies]

    # 3×1 vertical layout — one subplot per level, shared x-axis (strategy names)
    fig_h = min(1.3 * n_models * n_lv + 1.2, 5.0)
    fig, axes = plt.subplots(n_lv, 1, figsize=(fig_width, fig_h),
                              squeeze=False, constrained_layout=True)

    im = None
    global_vmax = 0
    # first pass: find global vmax for consistent color scale
    for lv in active_levels:
        mre_key = f"{lv}_mre"
        for r in rows:
            v = _to_float(r.get(mre_key, 0)) * 100
            if v > global_vmax:
                global_vmax = v
    global_vmax = max(global_vmax, 15)

    for li, lv in enumerate(active_levels):
        ax = axes[li][0]
        mre_key = f"{lv}_mre"
        arr = np.zeros((n_models, n_strats))
        for r in rows:
            m = r.get("model", "")
            s = r.get("strategy", "")
            if m in models and s in valid_strategies:
                mi = models.index(m)
                si = valid_strategies.index(s)
                arr[mi, si] = _to_float(r.get(mre_key, 0)) * 100

        im = ax.imshow(arr, cmap="YlOrRd", aspect="auto",
                        vmin=0, vmax=global_vmax)

        ax.set_xticks(np.arange(n_strats))
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(models)

        # only show x-tick labels on bottom subplot
        if li == n_lv - 1:
            ax.set_xticklabels(strat_names, rotation=35, ha="right")
        else:
            ax.set_xticklabels([])

        # cell annotations — dark text on light bg, white on dark bg
        for i in range(n_models):
            for j in range(n_strats):
                v = arr[i, j]
                if v > 0:
                    color = "white" if v > global_vmax * 0.65 else "black"
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                            fontsize=ANNOT_SIZE, fontweight="bold", color=color)
                else:
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=ANNOT_SIZE, color="#AAA")

        label_c = chr(97 + li)
        ax.set_title(f"({label_c}) {level_display.get(lv, lv)}", pad=4)

    # shared colorbar on the right
    if im is not None:
        cb = fig.colorbar(im, ax=[axes[i][0] for i in range(n_lv)],
                          shrink=0.8, pad=0.03)
        cb.set_label("MRE (%)")

    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=ANNOT_SIZE,
                 fontstyle="italic", color="#666")
    return fig


# ═══════════════════════════════════════════════════════════════════
#  Legacy aliases for backward compatibility
# ═══════════════════════════════════════════════════════════════════
plot_f3_overview = plot_f2_strategy_overview
plot_f4_pareto = plot_f2_strategy_overview
plot_f5_peak_comparison = plot_f3_peak_comparison
plot_f6_mre = plot_f4_mre
plot_f7_batch_scaling = None
plot_f8_heatmap = None
plot_f9_waterfall = None
plot_f2_timeline = None
