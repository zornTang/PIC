#!/usr/bin/env python3
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#2B6C9F",
    "orange": "#D9822B",
    "teal": "#2B9A8B",
    "gray": "#A9B0B8",
    "dark": "#1F2933",
    "grid": "#D7DDE3",
}


def apply_npj_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.35,
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=2.5, pad=1.5)


def emphasize_panel(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8.5, width=1.1, length=3.2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(9.5)
    ax.yaxis.label.set_fontsize(9.5)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontsize(10.0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.1)
