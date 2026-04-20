"""Publication-quality figure style for academic papers.

Usage::

    from toolkit.output.pub_style import paper_style, COLORS, LEVEL_COLORS, GROUP_COLORS

    with paper_style():
        fig, ax = plt.subplots(figsize=(COLUMN_SINGLE, 2.5))
        ...
        savefig_pub(fig, "output_dir/fig_name")   # saves .pdf + .png
"""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Column widths (inches) ─────────────────────────────────────────
COLUMN_SINGLE = 3.33   # 84.6 mm — ACM/IEEE single column
COLUMN_DOUBLE = 6.89   # 175 mm — full width

# ── Font sizes (pt) ───────────────────────────────────────────────
FONT_SIZE    = 8
LABEL_SIZE   = 8
TITLE_SIZE   = 9
LEGEND_SIZE  = 7
TICK_SIZE    = 7

# ── Colorblind-safe palette (Tol Qualitative) ─────────────────────
COLORS = {
    "blue":   "#4477AA",
    "orange": "#EE6677",
    "green":  "#228833",
    "purple": "#AA3377",
    "cyan":   "#66CCEE",
    "gray":   "#BBBBBB",
    "yellow": "#CCBB44",
}

# Simulation levels
LEVEL_COLORS = {
    "RT":   "#2D3436",
    "L1":   "#CCBB44",
    "L2":   "#EE6677",
    "L2.5": "#AA3377",
    "L3":   "#228833",
}

# Level hatching patterns (for B&W printing)
LEVEL_HATCHES = {
    "RT":   "",
    "L2":   "///",
    "L2.5": "\\\\\\",
    "L3":   "xxx",
}

# Strategy groups
GROUP_COLORS = {
    "G1": "#4477AA",
    "G2": "#EE6677",
    "G3": "#228833",
}
GROUP_LABELS = {
    "G1": "G1: Eager",
    "G2": "G2: Compiled",
    "G3": "G3: AC/SAC+Compiled",
}

# Memory composition
COMP_COLORS = {
    "param":      "#4477AA",
    "grad":       "#EE6677",
    "optimizer":  "#228833",
    "activation": "#AA3377",
}


# ── rcParams for paper style ──────────────────────────────────────
_PAPER_RC = {
    "font.family":       "serif",
    "font.size":         FONT_SIZE,
    "axes.labelsize":    LABEL_SIZE,
    "axes.titlesize":    TITLE_SIZE,
    "axes.titleweight":  "bold",
    "legend.fontsize":   LEGEND_SIZE,
    "xtick.labelsize":   TICK_SIZE,
    "ytick.labelsize":   TICK_SIZE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.2,
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
    "axes.axisbelow":    True,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":   1.5,
    "lines.markersize":  5,
    "patch.linewidth":   0.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "legend.borderpad":   0.3,
    "legend.handlelength": 1.2,
}


@contextmanager
def paper_style():
    """Context manager that activates publication rcParams, restores on exit."""
    old = mpl.rcParams.copy()
    mpl.rcParams.update(_PAPER_RC)
    try:
        yield
    finally:
        mpl.rcParams.update(old)


def savefig_pub(fig, path_stem, formats=("pdf", "png")):
    """Save figure in multiple formats.

    Args:
        fig: matplotlib Figure.
        path_stem: path without extension, e.g. ``"outputs/fig1"``.
        formats: tuple of format strings.
    """
    from pathlib import Path
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(stem.with_suffix(f".{fmt}")))


# Human-readable display names for strategy labels
_DISPLAY_NAMES = {
    "eager_baseline":       "Eager",
    "classic_ac":           "Classic AC",
    "sac_save_matmuls":     "SAC (matmul)",
    "sac_recompute_all":    "SAC (recomp)",
    "aot_eager+default":    "AOT+Default",
    "aot_eager+min_cut":    "AOT+MinCut",
    "inductor(b=1.0)":      "Inductor b=1",
    "inductor(b=0.5)":      "Inductor b=.5",
    "inductor(b=0.0)":      "Inductor b=0",
    "ac+aot_eager(default)": "AC+AOT",
    "ac+inductor":          "AC+Inductor",
    "sac_mm+inductor":      "SAC+Inductor",
}


def short_strategy_name(name: str) -> str:
    """Convert strategy names to clean display labels."""
    raw = name.split(" ", 1)[1] if " " in name else name
    return _DISPLAY_NAMES.get(raw, raw)


MB = 1024 ** 2
