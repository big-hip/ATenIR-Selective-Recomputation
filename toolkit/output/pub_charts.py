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
    COLUMN_SINGLE, COLUMN_DOUBLE,
    COLORS, LEVEL_COLORS, GROUP_COLORS, GROUP_LABELS,
    COMP_COLORS, LEVEL_HATCHES,
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
def plot_f1_composition(rows: List[Row], fig_width: float = COLUMN_SINGLE) -> plt.Figure:
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

    fig, ax = plt.subplots(figsize=(fig_width, 0.6 * n + 1.0))
    y = np.arange(n)
    lefts = np.zeros(n)

    for idx, (cname, color) in enumerate(zip(comp_names, comp_cols)):
        vals = np.array([c[idx] for c in comps])
        ax.barh(y, vals, left=lefts, height=0.55,
                color=color, edgecolor="white", linewidth=0.5,
                label=cname)
        for j, (v, l) in enumerate(zip(vals, lefts)):
            total = sum(comps[j])
            pct = v / total * 100 if total else 0
            if pct > 8:
                ax.text(l + v / 2, j, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
        lefts += vals

    for j in range(n):
        total = sum(comps[j])
        ax.text(total + total * 0.015, j, f"{total:.1f} {unit_label}",
                ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"Memory ({unit_label})")
    ax.set_title("Memory Composition Breakdown")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5,
              borderpad=0.3, columnspacing=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
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
                              fig_width: float = COLUMN_DOUBLE,
                              subtitle: str = "") -> plt.Figure:
    """Two-panel figure: (a) grouped bar of true_peak, (b) Pareto scatter.

    Expected keys: strategy, group, rt_true_peak, rt_step_ms.
    """
    names = [short_strategy_name(r.get("strategy", "")) for r in rows]
    raw_peaks = [_to_float(r.get("rt_true_peak", 0)) for r in rows]
    unit_div, unit_label = _auto_unit(max(raw_peaks) if raw_peaks else 0)
    peaks = [v / unit_div for v in raw_peaks]
    times = [_to_float(r.get("rt_step_ms", 0)) for r in rows]
    groups = [r.get("group", "G1") for r in rows]
    colors_bar = [GROUP_COLORS.get(g, COLORS["gray"]) for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 3.4),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # (a) Bar chart
    x = np.arange(len(names))
    bars = ax1.bar(x, peaks, color=colors_bar, edgecolor="white",
                   linewidth=0.5, width=0.72)

    # baseline reference per group
    baseline_map = {}
    for r in rows:
        g = r.get("group", "G1")
        if g not in baseline_map:
            baseline_map[g] = _to_float(r.get("rt_true_peak", 0))
    if "G3" not in baseline_map and "G2" in baseline_map:
        baseline_map["G3"] = baseline_map["G2"]

    for g, bl_val in baseline_map.items():
        bl_u = bl_val / unit_div
        ax1.axhline(bl_u, color=GROUP_COLORS.get(g, "gray"),
                    ls="--", lw=0.7, alpha=0.35)

    # savings annotations
    ymax = max(peaks) if peaks else 1
    for i, (r, pk) in enumerate(zip(rows, peaks)):
        g = r.get("group", "G1")
        bl = baseline_map.get(g, 0) / unit_div
        if bl and abs(pk - bl) / bl > 0.01:
            pct = (bl - pk) / bl * 100
            sign = "\u2212" if pct > 0 else "+"   # proper minus
            ax1.text(i, pk + ymax * 0.018,
                     f"{sign}{abs(pct):.0f}%",
                     ha="center", va="bottom", fontsize=5,
                     fontweight="bold", color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=38, ha="right", fontsize=5.5)
    ax1.set_ylabel(f"Peak Memory ({unit_label})")
    ax1.set_title("(a) Strategy Comparison", fontsize=8)

    handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g])
               for g in ["G1", "G2", "G3"] if g in set(groups)]
    ax1.legend(handles=handles, loc="upper right", fontsize=5.5,
               framealpha=0.9)

    # (b) Pareto scatter — with collision-aware label placement
    for i, (t, m, g, nm) in enumerate(zip(times, peaks, groups, names)):
        c = GROUP_COLORS.get(g, COLORS["gray"])
        ax2.scatter(t, m, c=c, s=48, edgecolors="white", linewidths=0.4,
                    zorder=3)

    # place labels avoiding overlap via greedy vertical offset
    label_data = sorted(enumerate(zip(times, peaks, names)),
                        key=lambda x: (x[1][0], x[1][1]))
    placed = []  # (x_pt, y_pt) in axes coords
    for idx, (t, m, nm) in label_data:
        g = groups[idx]
        # try offsets: right, left, up, down
        offsets = [(5, 0), (-5, 0), (0, 6), (0, -8)]
        best_off = offsets[0]
        for ox, oy in offsets:
            cand_x, cand_y = t + ox * 0.5, m + oy * 0.05
            collision = False
            for px, py in placed:
                if abs(cand_x - px) < 30 and abs(cand_y - py) < 0.8:
                    collision = True
                    break
            if not collision:
                best_off = (ox, oy)
                break
        ax2.annotate(nm, (t, m), textcoords="offset points",
                     xytext=best_off, fontsize=4.5, color="#444",
                     arrowprops=dict(arrowstyle="-", color="#BBB",
                                     lw=0.4) if best_off[1] != 0 else None)
        placed.append((t + best_off[0] * 0.5, m + best_off[1] * 0.05))

    front = _pareto_front(times, peaks)
    if len(front) >= 2:
        ft = sorted([(times[i], peaks[i]) for i in front])
        ax2.plot([p[0] for p in ft], [p[1] for p in ft],
                 color=COLORS["gray"], ls="--", lw=0.9, alpha=0.4, zorder=1)

    ax2.set_xlabel("Step Time (ms)")
    ax2.set_ylabel(f"Peak Memory ({unit_label})")
    ax2.set_title("(b) Memory-Time Tradeoff", fontsize=8)

    if subtitle:
        fig.suptitle(subtitle, fontsize=6.5, fontstyle="italic",
                     color="#666", y=0.01, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1] if subtitle else None)
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F3: Multi-Level Peak Comparison  (grouped bar RT/L2/L2.5/L3)
# ═══════════════════════════════════════════════════════════════════
def plot_f3_peak_comparison(rows: List[Row],
                            fig_width: float = COLUMN_DOUBLE,
                            subtitle: str = "") -> plt.Figure:
    """Grouped bar: RT vs L2 vs L2.5 vs L3 true_peak for simulatable strategies.

    Expected keys: strategy, rt_true_peak, l2_true_peak,
    l25_true_peak (optional), l3_true_peak (optional), mre, l25_mre, l3_mre.
    """
    sim_rows = [r for r in rows if _to_float(r.get("l2_true_peak")) > 0]
    if not sim_rows:
        sim_rows = rows

    names = [short_strategy_name(r.get("strategy", "")) for r in sim_rows]
    n = len(names)

    levels = [
        ("RT",   "rt_true_peak",  None),
        ("L2",   "l2_true_peak",  "mre"),
        ("L2.5", "l25_true_peak", "l25_mre"),
        ("L3",   "l3_true_peak",  "l3_mre"),
    ]

    active = []
    for lbl, key, _ in levels:
        if any(_to_float(r.get(key)) > 0 for r in sim_rows):
            active.append((lbl, key, _))
    n_bars = max(len(active), 1)

    # auto unit
    all_peaks = [_to_float(r.get(k, 0)) for r in sim_rows
                 for _, k, _ in active]
    unit_div, unit_label = _auto_unit(max(all_peaks) if all_peaks else 0)

    # lighter hatching
    _light_hatches = {"RT": "", "L2": "//", "L2.5": "\\\\", "L3": "xx"}

    w = 0.72 / n_bars
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    x = np.arange(n)

    bar_data = {}  # (strategy_idx, level_label) -> (x_pos, val)
    for bi, (lbl, key, mre_key) in enumerate(active):
        offset = (bi - (n_bars - 1) / 2) * w
        vals = [_to_float(r.get(key, 0)) / unit_div for r in sim_rows]
        c = LEVEL_COLORS.get(lbl, COLORS["gray"])
        hatch = _light_hatches.get(lbl, "")
        ax.bar(x + offset, vals, w * 0.90, color=c, edgecolor="white",
               linewidth=0.4, label=lbl, hatch=hatch, alpha=0.88)

        for j in range(n):
            bar_data[(j, lbl)] = (x[j] + offset, vals[j])

    # MRE annotation — only on highest bar per strategy to avoid overlap
    for j, r in enumerate(sim_rows):
        best_lbl, best_mre, best_c = None, None, None
        for lbl, key, mre_key in active:
            if mre_key:
                mre_val = _to_float(r.get(mre_key))
                if mre_val > 0:
                    if best_mre is None or mre_val > best_mre:
                        best_lbl, best_mre, best_c = lbl, mre_val, \
                            LEVEL_COLORS.get(lbl, COLORS["gray"])
        # show all MRE as a compact multi-line label above RT bar
        mre_parts = []
        for lbl, key, mre_key in active:
            if mre_key:
                mv = _to_float(r.get(mre_key))
                if mv > 0:
                    c = LEVEL_COLORS.get(lbl, "#333")
                    mre_parts.append((lbl, mv, c))
        if mre_parts:
            # place above the tallest bar in this group
            max_val = max(bar_data[(j, lbl)][1] for lbl, _, _ in active
                          if (j, lbl) in bar_data)
            rt_x = bar_data[(j, "RT")][0] if (j, "RT") in bar_data \
                else x[j]
            txt = " / ".join(f"{mv*100:.0f}%" for _, mv, _ in mre_parts)
            ax.text(x[j], max_val + max(all_peaks) / unit_div * 0.02,
                    txt, ha="center", va="bottom", fontsize=4,
                    color="#555", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel(f"True Peak ({unit_label})")
    ax.set_title("Multi-Level Estimation vs. Runtime")
    ax.legend(fontsize=6, loc="upper right", ncol=n_bars, framealpha=0.9)

    if subtitle:
        fig.suptitle(subtitle, fontsize=6.5, fontstyle="italic",
                     color="#666", y=0.01, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1] if subtitle else None)
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F4: Simulation Accuracy — MRE  (grouped bar L2/L2.5/L3)
# ═══════════════════════════════════════════════════════════════════
def plot_f4_mre(rows: List[Row],
                fig_width: float = COLUMN_DOUBLE,
                subtitle: str = "") -> plt.Figure:
    """Grouped bar of MRE (%) for L2, L2.5, L3 per strategy.

    Expected keys: strategy, mre, l25_mre, l3_mre,
    direction, l25_direction, l3_direction (optional).
    """
    sim_rows = [r for r in rows if _to_float(r.get("mre")) > 0]
    if not sim_rows:
        sim_rows = rows

    names = [short_strategy_name(r.get("strategy", "")) for r in sim_rows]
    n = len(names)

    mre_specs = [
        ("L2",   "mre",     "direction"),
        ("L2.5", "l25_mre", "l25_direction"),
        ("L3",   "l3_mre",  "l3_direction"),
    ]

    active = [(lbl, k, dk) for lbl, k, dk in mre_specs
              if any(_to_float(r.get(k)) > 0 for r in sim_rows)]
    n_bars = max(len(active), 1)
    w = 0.72 / n_bars

    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    x = np.arange(n)

    all_bar_vals = {}  # (j, lbl) -> val
    avg_lines = []
    for bi, (lbl, key, dir_key) in enumerate(active):
        offset = (bi - (n_bars - 1) / 2) * w
        vals = [_to_float(r.get(key, 0)) * 100 for r in sim_rows]
        c = LEVEL_COLORS.get(lbl, COLORS["gray"])
        ax.bar(x + offset, vals, w * 0.88, color=c, edgecolor="white",
               linewidth=0.4, label=lbl, alpha=0.88)

        for j in range(n):
            all_bar_vals[(j, lbl)] = vals[j]

        nonzero = [v for v in vals if v > 0]
        if nonzero:
            avg = sum(nonzero) / len(nonzero)
            avg_lines.append((lbl, avg, c))
            ax.axhline(avg, color=c, ls=":", lw=0.7, alpha=0.4)

    # compact MRE labels: one label per strategy showing all levels
    for j, r in enumerate(sim_rows):
        parts = []
        for lbl, key, dir_key in active:
            v = all_bar_vals.get((j, lbl), 0)
            if v > 0:
                d = r.get(dir_key, "")
                sign = "+" if d == "over" else ("\u2212" if d == "under" else "")
                parts.append(f"{sign}{v:.0f}")
        if parts:
            max_v = max(all_bar_vals.get((j, lbl), 0) for lbl, _, _ in active)
            txt = "/".join(parts) + "%"
            ax.text(x[j], max_v + 0.6, txt,
                    ha="center", va="bottom", fontsize=4,
                    color="#444", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("MRE (%)")
    ax.set_title("Simulation Accuracy — Mean Relative Error")

    handles_main = [Patch(facecolor=LEVEL_COLORS.get(lbl, "gray"), label=lbl)
                    for lbl, _, _ in active]
    for lbl, avg, c in avg_lines:
        handles_main.append(plt.Line2D([0], [0], color=c, ls=":", lw=0.7,
                                       label=f"{lbl} avg {avg:.1f}%"))
    ax.legend(handles=handles_main, fontsize=5.5, loc="upper left",
              ncol=2, borderpad=0.3, framealpha=0.9)

    if subtitle:
        fig.suptitle(subtitle, fontsize=6.5, fontstyle="italic",
                     color="#666", y=0.01, va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1] if subtitle else None)
    return fig


# ═══════════════════════════════════════════════════════════════════
#  F5: Peak Phase Heatmap (batch × optimizer → peak_phase)
# ═══════════════════════════════════════════════════════════════════
PHASE_COLORS = {"FW": COLORS["blue"], "BW": COLORS["orange"], "OPT": COLORS["purple"]}
PHASE_CODE = {"FW": 0, "BW": 1, "OPT": 2}


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

    # custom colormap: FW=blue, BW=orange, OPT=purple, OOM=gray
    cmap = mcolors.ListedColormap([COLORS["blue"], COLORS["orange"],
                                   COLORS["purple"], "#AAAAAA"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(fig_width * 0.6, 0.5 * n_batch + 1.0))

    ax.imshow(phase_arr, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_opt))
    ax.set_yticks(np.arange(n_batch))
    ax.set_xticklabels(optimizers, fontsize=7)
    ax.set_yticklabels([str(b) for b in batches], fontsize=7)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Batch Size")

    # cell annotations: phase + peak value (or "OOM")
    for i in range(n_batch):
        for j in range(n_opt):
            ph = phase_labels[i][j]
            pk = peak_arr[i, j]
            if ph == "OOM":
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=6,
                        fontweight="bold", color="white")
            elif ph:
                pk_gb = pk / GB
                if pk_gb >= 1.0:
                    ax.text(j, i, f"{ph}\n{pk_gb:.1f}GB",
                            ha="center", va="center", fontsize=5.5,
                            fontweight="bold", color="white")
                else:
                    ax.text(j, i, f"{ph}\n{pk/MB:.0f}MB",
                            ha="center", va="center", fontsize=5.5,
                            fontweight="bold", color="white")

    handles = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in ["FW", "BW", "OPT"]]
    handles.append(Patch(facecolor="#AAAAAA", label="OOM"))
    ax.legend(handles=handles, loc="upper left", fontsize=6,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_title(f"Peak Phase — {strategy}", fontsize=8)
    fig.tight_layout()
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
    phase_colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]

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
                   color=pcol, edgecolor="white", linewidth=0.4,
                   label=plbl if oi == 0 else "")
            bottoms += vals_arr

        # annotate true_peak on top
        for bi, b in enumerate(batches):
            match = [r for r in strat_rows
                     if r.get("optimizer") == opt_name
                     and int(_to_float(r.get("batch", 0))) == b]
            if match:
                tp = _to_float(match[0].get("true_peak", 0)) / unit_div
                ph = match[0].get("peak_phase", "")
                ax.text(bi, bottoms[bi] + bottoms.max() * 0.02,
                        f"{tp:.1f}{unit_label}\n({ph})",
                        ha="center", va="bottom", fontsize=5,
                        fontweight="bold", color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches], fontsize=7)
        ax.set_xlabel("Batch Size")
        ax.set_title(opt_name, fontsize=8)

    axes[0].set_ylabel(f"Peak Memory ({unit_label})")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Three-Phase Peak Breakdown — {strategy}", fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
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
    im = ax.imshow(arr, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(n_strats))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([short_strategy_name(s) for s in valid_strategies],
                       rotation=35, ha="right", fontsize=6)
    ax.set_yticklabels(models, fontsize=7)

    for i in range(n_models):
        for j in range(n_strats):
            v = arr[i, j]
            d = dir_arr[i][j]
            if v > 0:
                arrow = "+" if d == "over" else ("-" if d == "under" else "")
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{arrow}{v:.1f}%",
                        ha="center", va="center",
                        fontsize=6, fontweight="bold", color=color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=5.5, color="#999")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cb.set_label("MRE (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    level_display = {"l2": "L2", "l25": "L2.5", "l3": "L3"}.get(level, level)
    ax.set_title(f"Model Generalization — {level_display} MRE", fontsize=8)
    fig.tight_layout()
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
