"""
Unified style configuration for all PIC visualization scripts.

Usage:
    from style_config import apply_style, PALETTE, FIGSIZE, CMAPS, style_ax, save_fig

Call apply_style() once at the top of each script (after imports, before any plotting).
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Color System
# ---------------------------------------------------------------------------

PALETTE = {
    # Primary group colors (Human vs Immune)
    "human":   "#2C5F8A",   # steel blue      — Human-level
    "immune":  "#D4622A",   # burnt orange    — Immune-level
    "common":  "#3A9E8F",   # teal            — Commonly Essential

    # Aliases for backward compatibility with existing scripts
    "primary":   "#2C5F8A",
    "secondary": "#D4622A",
    "tertiary":  "#3A9E8F",

    # Accent / semantic
    "sig":      "#B03030",   # significant / highlight (p-val, annotation)
    "neutral":  "#C8CDD6",   # non-essential / background points (light gray)
    "ref_line": "#A0A8B4",   # dashed reference lines (diagonal, baseline)
    "bg":       "#F5F7FA",   # panel background fill (use sparingly)
    "grid":     "#DDE1E8",   # grid lines

    # Performance curve colors (for ROC / PR / training curves)
    "roc":   "#2C5F8A",   # AUROC curve  — same as human
    "pr":    "#D4622A",   # AUPRC curve  — same as immune
    "train": "#2C5F8A",   # training loss/metric
    "val":   "#D4622A",   # validation loss/metric

    # Ablation palette (5 variants — perceptually uniform, print-safe)
    "abl": ["#3B4FA0", "#2980B9", "#27AE8F", "#E67E22", "#C0392B"],
}

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

CMAPS = {
    # Heatmaps: white → deep navy (for AA composition, subcellular, etc.)
    "heat_blue": LinearSegmentedColormap.from_list(
        "heat_blue",
        ["#F0F5FF", "#D1DFF4", "#A6C4EA", "#6F98D1", "#3F67B1", PALETTE["human"]],
    ),
    # Heatmaps: white → teal → orange (for enrichment bubble coloring)
    "bubble": LinearSegmentedColormap.from_list(
        "bubble",
        ["#EFF3FB", PALETTE["common"], PALETTE["immune"]],
    ),
    # Diverging: blue ↔ orange (for difference / delta plots)
    "diverge": LinearSegmentedColormap.from_list(
        "diverge",
        [PALETTE["human"], "#F7F8FA", PALETTE["immune"]],
    ),
    # Attention heatmap: white → deep navy
    "attention": LinearSegmentedColormap.from_list(
        "attention",
        ["#FFFFFF", "#C8D9F0", "#6F98D1", PALETTE["human"]],
    ),
}

# ---------------------------------------------------------------------------
# Figure Sizes  (width, height) in inches
# ---------------------------------------------------------------------------

FIGSIZE = {
    "single":      (6.0, 4.2),    # single-panel, single column
    "wide":        (10.0, 4.2),   # wide single-panel (e.g. scatter)
    "double":      (13.5, 5.2),   # two panels side-by-side
    "quad":        (13.5, 10.5),  # 2×2 panel (physicochem, training curves)
    "tall":        (5.0, 8.0),    # tall single-column (subcellular heatmap)
    "bubble_col":  (9.0, 4.8),    # one group in a bubble column plot
    "attention":   (14.0, 5.0),   # single-protein attention dense figure
    "aa_heat":     (11.0, 4.8),   # amino acid composition heatmap
}

# ---------------------------------------------------------------------------
# Core rcParams
# ---------------------------------------------------------------------------

BASE_RC = {
    # Resolution
    "figure.dpi":   150,
    "savefig.dpi":  600,

    # Font
    "font.size":          12,
    "font.sans-serif":    ["Arial", "DejaVu Sans", "Liberation Sans",
                           "Bitstream Vera Sans", "sans-serif"],
    "font.family":        "sans-serif",
    "axes.unicode_minus": False,

    # Title / label sizes
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   13,
    "axes.labelweight": "bold",

    # Tick sizes
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.minor.size":  3,
    "ytick.minor.size":  3,

    # Spine / frame
    "axes.linewidth":  2.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,

    # Grid
    "axes.grid":       True,
    "grid.color":      "#DDE1E8",
    "grid.linestyle":  "--",
    "grid.linewidth":  0.8,
    "grid.alpha":      0.55,
    "axes.axisbelow":  True,

    # Legend
    "legend.fontsize":       11,
    "legend.title_fontsize": 11,
    "legend.framealpha":     0.88,
    "legend.edgecolor":      "#C8CDD6",
    "legend.borderpad":      0.5,

    # Lines / markers
    "lines.linewidth":  2.5,
    "lines.markersize": 4.5,

    # Save
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.08,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply the unified rcParams. Call once per script before any plotting."""
    plt.rcParams.update(BASE_RC)


def style_ax(ax: "mpl.axes.Axes", *, bold_ticks: bool = True) -> None:
    """
    Polish an Axes object:
    - enforce spine linewidth
    - optionally bold tick labels
    Top/right spines are already hidden via rcParams.
    """
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(width=2.0, length=5)
    if bold_ticks:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                lbl.set_fontweight("bold")
            except Exception:
                pass


def panel_label(ax: "mpl.axes.Axes", label: str, size: float = 14.0):
    """Add a bold panel label (e.g. 'A', 'B') to the top-left of an Axes."""
    return ax.text(
        -0.15, 1.06, label,
        transform=ax.transAxes,
        fontsize=size, fontweight="bold",
        va="bottom", ha="left",
    )


def save_fig(
    fig: "mpl.figure.Figure",
    path: str,
    *,
    pdf: bool = True,
    png: bool = True,
) -> None:
    """
    Save figure as PDF and/or PNG.
    Creates parent directory if needed.
    PDF is suitable for LaTeX inclusion; PNG is for preview.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base, _ = os.path.splitext(path)
    if pdf:
        fig.savefig(base + ".pdf")
    if png:
        fig.savefig(base + ".png", facecolor="none", transparent=True)
    plt.close(fig)
