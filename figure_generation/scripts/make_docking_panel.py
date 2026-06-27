#!/usr/bin/env python3
"""
Docking results panel figure for thesis Chapter 4.

Uses dock_out_p2r dataset (pH 4.5 protonated receptor, 10 ligands).
  (a) – Horizontal bar chart: ranked binding affinities for all ligands
  (b) – Scatter: best affinity vs Cys209 SG distance

Output: result/docking_panel/docking_panel_tj.{pdf,png}
        mythesis/figures/docking_panel.pdf
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _fg_paths import FIGURES_NPJ_DIR, PIC2_DOCKING_DIR
from style_config import apply_style, PALETTE, panel_label

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
DATA_CSV = PIC2_DOCKING_DIR / "docking_summary.csv"
NPJ_OUT  = FIGURES_NPJ_DIR

# ---------------------------------------------------------------------------
# Data & display names
# ---------------------------------------------------------------------------
DISPLAY = {
    "Gly-AMC":       "Gly-AMC",
    "LLS":           "LL-Ser",
    "LR":            "Leu-Arg",
    "LLG":           "LL-Gly",
    "LDTT_baseline": "LDTT",
    "LDTT_focus":    "LDTT (focus)",
    "Pro-AMC":       "Pro-AMC",
    "IDTT_baseline": "IDTT",
    "Leu-AMC":       "Leu-AMC",
    "LLL-AMC":       "LLL-AMC",
}

AMC_SUBS = {"Gly-AMC", "Leu-AMC", "LLL-AMC", "Pro-AMC"}
PEP_SUBS = {"LLG", "LLS", "LR"}
CONTROLS = {"IDTT_baseline", "LDTT_baseline", "LDTT_focus"}

# Colors align with unified palette
CAT_COLOR = {
    "AMC substrate": PALETTE["human"],    # steel blue
    "Peptide":       PALETTE["common"],   # teal
    "Control":       PALETTE["neutral"],  # light gray
}


def load_data(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lig = row["ligand"]
            def pf(k):
                try:
                    return float(row[k])
                except (ValueError, KeyError):
                    return math.nan
            cys = pf("cys209_distance")
            rows.append({
                "ligand":    lig,
                "label":     DISPLAY.get(lig, lig),
                "best":      pf("best_affinity"),
                "mean_top3": pf("mean_top3"),
                "cys_dist":  cys,
                "category": (
                    "AMC substrate" if lig in AMC_SUBS else
                    "Peptide"       if lig in PEP_SUBS else
                    "Control"
                ),
            })
    # weakest (least negative) at bottom (y=0), strongest at top (y=N-1)
    rows.sort(key=lambda r: r["best"], reverse=True)
    return rows


def panel_a_bars(ax: plt.Axes, rows: list[dict]) -> None:
    """(a) Horizontal bar chart of best docking affinity, ranked strongest→top."""
    labels  = [r["label"]    for r in rows]
    best    = [r["best"]     for r in rows]
    mean_t3 = [r["mean_top3"] for r in rows]
    colors  = [CAT_COLOR[r["category"]] for r in rows]

    y     = np.arange(len(rows))
    bar_h = 0.55

    # Bars
    bars = ax.barh(y, best, height=bar_h, color=colors,
                   edgecolor="white", linewidth=1.0, zorder=3)

    # Whisker: best → mean_top3 (pose stability)
    for i, (b, m) in enumerate(zip(best, mean_t3)):
        if not math.isnan(m) and b != m:
            ax.plot([b, m], [y[i], y[i]], color=PALETTE["ref_line"],
                    lw=2.2, solid_capstyle="round", zorder=4)
            ax.plot(m, y[i], "d", color=PALETTE["ref_line"], ms=4.4, zorder=5)

    x_min = min(best) - 0.9   # extra room for labels outside bar ends

    # Value labels — to the LEFT of each bar's left tip (outside, never obstructs bar)
    for bar, val in zip(bars, best):
        ax.text(val - 0.08, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="right",
                fontsize=11, fontweight="bold", color="#444444", zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Best binding affinity (kcal/mol)", fontsize=15, fontweight="bold")
    ax.set_xlim(x_min, 0.4)
    ax.axvline(0, color=PALETTE["ref_line"], lw=1.6, ls="--", zorder=2)
    ax.grid(axis="x", zorder=1, linestyle="--", alpha=0.22, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=13, width=2.2, length=6)
    ax.tick_params(axis='y', labelsize=12, width=2.2, length=4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.3)

    # Panel label (a) — consistent with other thesis figures
    panel_label(ax, "(a)")
    ax.set_title("")

    # Legend — below x-axis label, horizontal, never overlaps bars
    patches = [mpatches.Patch(facecolor=c, label=lbl, edgecolor="white")
               for lbl, c in CAT_COLOR.items()]
    ax.legend(handles=patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=11, framealpha=0.0, edgecolor="none")


def panel_b_scatter(ax: plt.Axes, rows: list[dict]) -> None:
    """(b) Scatter: Cys209 SG distance vs best affinity."""
    CYS_THRESH = 3.6  # Å

    rng = np.random.default_rng(42)

    # Labels only for informative points; offsets tuned to avoid overlap
    label_cfg = {
        "LLL-AMC":       ( 0.30,  0.00, "left"),
        "Gly-AMC":       (-0.40,  0.05, "right"),
        "Leu-AMC":       ( 0.30,  0.00, "left"),
        "LLS":           ( 0.30, -0.05, "left"),
        "IDTT_baseline": ( 0.30,  0.05, "left"),
        "Pro-AMC":       ( 0.30,  0.00, "left"),
    }

    for r in rows:
        if math.isnan(r["cys_dist"]) or math.isnan(r["best"]):
            continue
        c    = CAT_COLOR[r["category"]]
        # edge same hue as fill but slightly darker for depth
        edge = {
            PALETTE["human"]:   "#1A3F5C",
            PALETTE["common"]:  "#226B5F",
            PALETTE["neutral"]: "#888888",
        }.get(c, "#444444")
        x    = r["cys_dist"] + rng.uniform(-0.04, 0.04)
        y    = r["best"]
        ax.scatter(x, y, color=c, s=75, edgecolors=edge,
                   linewidths=1.2, zorder=4)

        if r["ligand"] in label_cfg:
            ox, oy, ha = label_cfg[r["ligand"]]
            ax.text(x + ox, y + oy, r["label"],
                    fontsize=10.5, fontweight="bold", ha=ha, va="center", color="#333333")

    # Threshold line
    ax.axvline(CYS_THRESH, color=PALETTE["immune"], ls="--", lw=2.2, zorder=3,
               label=f"Cys209 threshold ({CYS_THRESH} Å)")

    # Shading: ≤ 3.6 Å zone
    ax.axvspan(0, CYS_THRESH, alpha=0.06, color=PALETTE["common"], zorder=1)

    ax.set_xlim(0, 13.5)
    ax.set_xlabel("Min. distance to Cys209 SG (Å)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Best binding affinity (kcal/mol)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=1,
              framealpha=0.0, edgecolor="none")
    ax.grid(True, zorder=0, linestyle="--", alpha=0.22, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=13, width=2.2, length=6)
    ax.tick_params(axis='y', labelsize=13, width=2.2, length=6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.3)

    # Panel label (b)
    panel_label(ax, "(b)")
    ax.set_title("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = load_data(DATA_CSV)
    if not rows:
        raise SystemExit(f"No data in {DATA_CSV}")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.44, left=0.13, right=0.97,
                        top=0.90, bottom=0.14)

    panel_a_bars(axes[0], rows)
    panel_b_scatter(axes[1], rows)

    NPJ_OUT.mkdir(parents=True, exist_ok=True)
    plt.close(fig)

    single_specs = [
        ("fig_17a_docking_affinity_bars.png", panel_a_bars, (7.0, 5.2)),
        ("fig_17b_docking_affinity_vs_distance.png", panel_b_scatter, (6.4, 5.2)),
    ]
    for filename, panel_fn, figsize in single_specs:
        sf, sa = plt.subplots(figsize=figsize)
        panel_fn(sa, rows)
        sa.set_title("")
        for text in list(sa.texts):
            if text.get_text() in {"(a)", "(b)"}:
                text.remove()
        legend = sa.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight('bold')
        sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
        plt.close(sf)
    print(f"Saved 17a/17b docking panels to {NPJ_OUT}")


if __name__ == "__main__":
    main()
