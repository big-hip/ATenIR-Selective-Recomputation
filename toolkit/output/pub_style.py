"""Publication-quality figure style for academic papers.

Style inspired by SciencePlots + Nature guidelines:
  - Sans-serif font (Arial / Helvetica)
  - Inward-pointing ticks
  - No frame on legends
  - Okabe-Ito colorblind-safe palette (Nature Methods recommended)
  - No global grid (add per-figure where helpful)

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
COLUMN_THESIS = 5.9    # ~150 mm — A4 single-column thesis body width

# ── Font sizes (pt) ───────────────────────────────────────────────
FONT_SIZE    = 8
LABEL_SIZE   = 9
TITLE_SIZE   = 10
LEGEND_SIZE  = 7.5
TICK_SIZE    = 8
ANNOT_SIZE   = 8        # minimum annotation / data-label size (≥8pt per v3)

# ── Okabe-Ito palette — Nature Methods recommended, colorblind-safe ─
COLORS = {
    "blue":   "#0072B2",   # deep blue
    "orange": "#E69F00",   # orange
    "red":    "#D55E00",   # vermillion
    "green":  "#009E73",   # bluish green
    "purple": "#CC79A7",   # reddish purple / pink
    "cyan":   "#56B4E9",   # sky blue
    "gray":   "#BDBDBD",   # neutral gray
    "gold":   "#F0E442",   # yellow
    "black":  "#000000",   # black
}

# Simulation levels — consistent triad across F3/F4
LEVEL_COLORS = {
    "RT":   "#555555",   # neutral dark gray — ground truth
    "L1":   "#F0E442",
    "L2":   "#0072B2",   # deep blue
    "L2.5": "#D55E00",   # vermillion
    "L3":   "#009E73",   # green
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
    "G1": "#0072B2",
    "G2": "#D55E00",
    "G3": "#009E73",
}
GROUP_LABELS = {
    "G1": "G1: Eager",
    "G2": "G2: Compiled",
    "G3": "G3: AC/SAC+Compiled",
}

# Memory composition
COMP_COLORS = {
    "param":      "#0072B2",
    "grad":       "#D55E00",
    "optimizer":  "#009E73",
    "activation": "#56B4E9",
}

# Per-model colours
MODEL_COLORS = {
    "gpt2":    "#0072B2",
    "llama":   "#D55E00",
    "mistral": "#009E73",
}


# ── rcParams for paper style ──────────────────────────────────────
_PAPER_RC = {
    # Font — sans-serif per Nature / modern CS venues
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          FONT_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "axes.titlesize":     TITLE_SIZE,
    "axes.titleweight":   "bold",
    "legend.fontsize":    LEGEND_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    # Ticks — inward (professional convention)
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.5,
    "xtick.major.width":  0.6,
    "ytick.major.size":   3.5,
    "ytick.major.width":  0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.pad":    4,
    "ytick.major.pad":    4,
    # Spines — only left & bottom
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    # Grid — OFF by default (add per-figure where needed)
    "axes.grid":          False,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.4,
    "grid.linestyle":     "-",
    "axes.axisbelow":     True,
    # Lines & markers
    "lines.linewidth":    1.2,
    "lines.markersize":   5,
    "patch.linewidth":    0.5,
    # Legend — frameless, clean
    "legend.frameon":     False,
    "legend.borderpad":   0.4,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    # Save
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
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


def savefig_pub(fig, path_stem, formats=("png",)):
    """Save figure in multiple formats, then close it to free memory.

    Args:
        fig: matplotlib Figure.
        path_stem: path without extension, e.g. ``"outputs/fig1"``.
        formats: tuple of format strings.
    """
    import matplotlib.pyplot as _plt
    from pathlib import Path
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(stem.with_suffix(f".{fmt}")))
    _plt.close(fig)


# Short display names (≤6 chars) for vertical bar chart labels
_DISPLAY_NAMES = {
    "eager_baseline":       "Eager",
    "classic_ac":           "C-AC",
    "sac_save_matmuls":     "SAC-M",
    "sac_recompute_all":    "SAC-R",
    "aot_eager+default":    "AOT-D",
    "aot_eager+min_cut":    "AOT-M",
    "inductor(b=1.0)":      "IND-1",
    "inductor(b=0.5)":      "IND-.5",
    "inductor(b=0.0)":      "IND-0",
    "ac+aot_eager(default)": "AC+A",
    "ac+inductor":          "AC+I",
    "sac_mm+inductor":      "SAC+I",
}


def short_strategy_name(name: str) -> str:
    """Convert strategy names to clean display labels."""
    raw = name.split(" ", 1)[1] if " " in name else name
    return _DISPLAY_NAMES.get(raw, raw)


MB = 1024 ** 2

# Marker styles for parity scatter (F3)
LEVEL_MARKERS = {
    "L2":   "o",
    "L2.5": "s",
    "L3":   "D",
}
