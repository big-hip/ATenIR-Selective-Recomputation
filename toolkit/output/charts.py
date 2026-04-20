import dataclasses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from toolkit.utils import format_bytes

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_BYTES_FORMATTER = FuncFormatter(lambda x, _: format_bytes(int(x)))

# 7 timeline sample points in order
_TIMELINE_KEYS = ["base", "fw_peak", "after_fw", "bw_peak", "after_bw", "opt_peak", "after_opt"]
_TIMELINE_LABELS = ["Base", "FW Peak", "After FW", "BW Peak", "After BW", "OPT Peak", "After OPT"]
_PEAK_INDICES = {1, 3, 5}  # indices of peak points in _TIMELINE_KEYS

# Colors
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_PHASE_COLORS = {"fw_peak": "#1f77b4", "bw_peak": "#ff7f0e", "opt_peak": "#d62728"}


def _to_dict(item):
    """Convert PhaseResult/dataclass to dict, or return dict as-is."""
    if isinstance(item, dict):
        return item
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        d = dataclasses.asdict(item)
        d.setdefault("base", d.get("base_allocated", 0))
        return d
    return vars(item)


def bar_chart_memory(results, save_path=None):
    names = []
    peaks = []
    for result in results:
        names.append(result.get("name") or result.get("tag") or result.get("strategy") or str(len(names)))
        peaks.append(result.get("peak_allocated") or result.get("estimated_peak") or result.get("runtime_peak") or 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, peaks)
    ax.set_ylabel("Bytes")
    ax.set_title("Memory Comparison")
    ax.yaxis.set_major_formatter(_BYTES_FORMATTER)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def line_chart_mre(mre_list, save_path=None):
    names = []
    values = []
    for item in mre_list:
        names.append(item.get("model") or item.get("name") or str(len(names)))
        values.append(item.get("mre_allocated") or item.get("mre") or item.get("l2_mre") or 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(names, values, marker="o")
    ax.set_ylabel("MRE")
    ax.set_title("MRE Trend")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def phase_timeline_chart(items, save_path=None):
    """Training step memory timeline: 7 sample points per strategy/source."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(_TIMELINE_KEYS))

    for i, item in enumerate(items):
        d = _to_dict(item)
        name = d.get("name") or d.get("tag") or f"Strategy {i}"
        vals = [d.get(k, 0) for k in _TIMELINE_KEYS]
        color = _COLORS[i % len(_COLORS)]
        ax.plot(x, vals, marker="o", color=color, label=name, linewidth=2, markersize=6)
        # Highlight peak points with larger markers
        for pi in _PEAK_INDICES:
            if vals[pi] > 0:
                ax.plot(pi, vals[pi], marker="D", color=color, markersize=10, zorder=5)
                ax.annotate(format_bytes(int(vals[pi])),
                            (pi, vals[pi]), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(_TIMELINE_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Memory")
    ax.set_title("Training Step Memory Timeline")
    ax.yaxis.set_major_formatter(_BYTES_FORMATTER)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def phase_grouped_bar(items, save_path=None):
    """Grouped bar chart: fw_peak / bw_peak / opt_peak side by side."""
    phases = ["fw_peak", "bw_peak", "opt_peak"]
    phase_labels = ["FW Peak", "BW Peak", "OPT Peak"]
    n_items = len(items)
    n_phases = len(phases)
    width = 0.8 / n_phases
    x = np.arange(n_items)

    fig, ax = plt.subplots(figsize=(max(8, n_items * 2), 5))
    for j, (phase, plabel) in enumerate(zip(phases, phase_labels)):
        vals = []
        for item in items:
            d = _to_dict(item)
            vals.append(d.get(phase, 0))
        offset = (j - n_phases / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=plabel, color=_PHASE_COLORS[phase])

    names = []
    for item in items:
        d = _to_dict(item)
        names.append(d.get("name") or d.get("tag") or d.get("strategy") or "")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Memory")
    ax.set_title("Phase Peak Comparison")
    ax.yaxis.set_major_formatter(_BYTES_FORMATTER)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def savings_waterfall(items, baseline_index=0, metric="true_peak", save_path=None):
    """Horizontal bar showing savings % relative to baseline."""
    dicts = [_to_dict(item) for item in items]
    base_val = dicts[baseline_index].get(metric, 0) or dicts[baseline_index].get("estimated_peak", 0)
    if base_val == 0:
        return plt.figure()

    names = []
    savings = []
    for i, d in enumerate(dicts):
        if i == baseline_index:
            continue
        val = d.get(metric, 0) or d.get("estimated_peak", 0)
        pct = (base_val - val) / base_val * 100
        names.append(d.get("name") or d.get("tag") or d.get("strategy") or f"#{i}")
        savings.append(pct)

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in savings]
    y = np.arange(len(names))
    ax.barh(y, savings, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Savings (%)")
    ax.set_title(f"Memory Savings vs Baseline ({metric})")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, s in enumerate(savings):
        ax.text(s + (1 if s >= 0 else -1), i, f"{s:+.1f}%", va="center",
                fontsize=9, color=colors[i])
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def stacked_breakdown(items, save_path=None):
    """Stacked bar: param / grad / optim / activation breakdown."""
    categories = []
    param_vals, grad_vals, optim_vals, act_vals = [], [], [], []
    for item in items:
        d = _to_dict(item)
        categories.append(d.get("name") or d.get("tag") or "")
        param_vals.append(d.get("param_bytes", 0))
        grad_vals.append(d.get("grad_bytes", 0))
        optim_vals.append(d.get("optimizer_bytes", d.get("optim_bytes", 0)))
        act_vals.append(d.get("activation_bytes", d.get("act_peak",
                        max(0, d.get("fwbw_peak", 0) - d.get("base", 0)))))

    x = np.arange(len(categories))
    width = 0.5
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 2), 5))

    p = np.array(param_vals)
    g = np.array(grad_vals)
    o = np.array(optim_vals)
    a = np.array(act_vals)

    ax.bar(x, p, width, label="Param", color="#1f77b4")
    ax.bar(x, g, width, bottom=p, label="Grad", color="#ff7f0e")
    ax.bar(x, o, width, bottom=p + g, label="Optimizer", color="#2ca02c")
    ax.bar(x, a, width, bottom=p + g + o, label="Activation", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Memory")
    ax.set_title("Memory Composition Breakdown")
    ax.yaxis.set_major_formatter(_BYTES_FORMATTER)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def heatmap_strategy_model(data, save_path=None):
    try:
        values = data.values
        row_labels = list(data.index)
        col_labels = list(data.columns)
    except Exception:
        values = data
        row_labels = [str(i) for i in range(len(data))]
        col_labels = [str(i) for i in range(len(data[0]) if data else 0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title("Strategy vs Model")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig
