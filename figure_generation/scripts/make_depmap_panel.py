#!/usr/bin/env python3
"""
DepMap ATP6V1B2 dependency panel for thesis Chapter 4.

Four-panel figure in unified style:
  (a) Violin + box: Chronos score distribution by lineage group
  (b) CDF: cumulative distribution showing myeloid left-shift
  (c) Tail enrichment: OR & fraction in extreme-dependency tails
  (d) Top myeloid models: horizontal bar, colored by disease subtype

Output: mythesis/figures/atp6v1b2_depmap_thesis_main.pdf / .png
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _fg_paths import DEP_MAP_TABLES, FIGURES_NPJ_DIR
from style_config import apply_style, PALETTE, panel_label

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TABLES = DEP_MAP_TABLES
NPJ_OUT = FIGURES_NPJ_DIR

# ---------------------------------------------------------------------------
# Colors — map to unified palette
# ---------------------------------------------------------------------------
GROUP_COLOR = {
    "Myeloid":    PALETTE["human"],    # steel blue  — key finding group
    "Lymphoid":   PALETTE["common"],   # teal        — comparison
    "Non-immune": PALETTE["neutral"],  # light gray  — background
}
DISEASE_COLOR = {
    "AML":          PALETTE["human"],    # steel blue
    "MPN/CML":      PALETTE["immune"],   # burnt orange
    "Other myeloid": PALETTE["common"],  # teal
}
TAIL_COLOR = {
    "Top 5% tail":  PALETTE["human"],   # steel blue  — stricter threshold
    "Top 15% tail": PALETTE["common"],  # teal        — broader threshold
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pf(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return math.nan


# ---------------------------------------------------------------------------
# Panel (a): violin + box
# ---------------------------------------------------------------------------
def panel_a_violin(ax: plt.Axes, scores: list[dict], *, show_title: bool = True, show_label: bool = True) -> None:
    groups   = ["Myeloid", "Lymphoid", "Non-immune"]
    data     = {g: [pf(r["ATP6V1B2"]) for r in scores if r["DependencyGroup"] == g]
                for g in groups}
    ns       = {g: len(data[g]) for g in groups}
    medians  = {g: float(np.median(data[g])) for g in groups}

    positions = np.arange(len(groups))
    vparts = ax.violinplot([data[g] for g in groups], positions=positions,
                           widths=0.7, showmedians=False, showextrema=False)

    for i, (g, body) in enumerate(zip(groups, vparts["bodies"])):
        body.set_facecolor(GROUP_COLOR[g])
        body.set_alpha(0.38)
        body.set_edgecolor(GROUP_COLOR[g])
        body.set_linewidth(2.3)

    # Box overlay
    for i, g in enumerate(groups):
        d = np.array(data[g])
        q1, q3 = np.percentile(d, [25, 75])
        med = np.median(d)
        ax.plot([i, i], [q1, q3], color=GROUP_COLOR[g], lw=6.8, solid_capstyle="round", zorder=3)
        ax.plot(i, med, "o", color="white", ms=8.2, zorder=4,
                markeredgecolor=GROUP_COLOR[g], markeredgewidth=2.0)

    # Threshold lines
    ax.axhline(-1.0, color=PALETTE["immune"], ls="--", lw=2.4, zorder=2,
               label="strong dep. (≤ −1.0)")
    ax.axhline(-0.5, color=PALETTE["ref_line"], ls=":", lw=2.1, zorder=2,
               label="dep. (≤ −0.5)")

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=14.5)
    ax.set_ylabel("ATP6V1B2 Chronos score", fontsize=15.5, fontweight="bold")
    ax.set_ylim(-3.9, 0.6)
    leg = ax.legend(fontsize=12.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                    ncol=2, framealpha=0.0, edgecolor="none")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if show_title:
        ax.set_title("Lineage-level distribution", loc="left", fontsize=15.0, fontweight="bold")
    if show_label:
        panel_label(ax, "(a)", size=16.0)

    # n & median annotations below x-tick labels
    for i, g in enumerate(groups):
        ax.text(i, -3.75, f"n={ns[g]},  med={medians[g]:.2f}",
                ha="center", va="center", fontsize=11.5,
                color=GROUP_COLOR[g], fontweight="bold")


# ---------------------------------------------------------------------------
# Panel (b): CDF
# ---------------------------------------------------------------------------
def panel_b_cdf(ax: plt.Axes, scores: list[dict], *, show_title: bool = True, show_label: bool = True) -> None:
    groups = ["Myeloid", "Lymphoid", "Non-immune"]
    data   = {g: sorted([pf(r["ATP6V1B2"]) for r in scores if r["DependencyGroup"] == g])
              for g in groups}

    for g in groups:
        d = np.array(data[g])
        n = len(d)
        ax.plot(d, np.arange(1, n + 1) / n,
                color=GROUP_COLOR[g], lw=3.6,
                label=g, zorder=3 if g == "Myeloid" else 2)

    ax.axvline(-1.0, color=PALETTE["immune"], ls="--", lw=2.4, zorder=4)
    ax.text(-1.05, 0.05, "≤ −1.0", ha="right", fontsize=12.5,
            color=PALETTE["immune"], fontweight="bold")

    ax.set_xlabel("ATP6V1B2 Chronos score", fontsize=15.5, fontweight="bold")
    ax.set_ylabel("Cumulative fraction", fontsize=15.5, fontweight="bold")
    ax.set_xlim(-3.8, 0.3)
    ax.set_ylim(0, 1.02)
    leg = ax.legend(fontsize=12.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                    ncol=3, framealpha=0.0, edgecolor="none")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if show_title:
        ax.set_title("Uniformity of strong dependency", loc="left",
                     fontsize=15.0, fontweight="bold")
    if show_label:
        panel_label(ax, "(b)", size=16.0)


# ---------------------------------------------------------------------------
# Panel (c): tail enrichment bars
# ---------------------------------------------------------------------------
def panel_c_enrichment(ax: plt.Axes, tail_rows: list[dict], *, show_title: bool = True, show_label: bool = True) -> None:
    enrich_groups = ["Myeloid", "AML/MPN-like", "Lymphoid"]
    tail_labels   = ["Top 5% tail", "Top 15% tail"]
    n_groups      = len(enrich_groups)
    n_tails       = len(tail_labels)
    bar_w         = 0.32
    x             = np.arange(n_groups)

    # Background fraction (same for all within a tail level)
    bg = {}
    for row in tail_rows:
        tl = row["tail_label"]
        if tl not in bg:
            bg[tl] = pf(row["background_fraction"])

    for ti, tl in enumerate(tail_labels):
        offset = (ti - 0.5) * bar_w
        for gi, grp in enumerate(enrich_groups):
            match = [r for r in tail_rows if r["tail_label"] == tl and r["group"] == grp]
            if not match:
                continue
            r = match[0]
            frac = pf(r["tail_fraction"])
            OR   = pf(r["odds_ratio"])
            pval = pf(r["pvalue"])
            col  = TAIL_COLOR[tl]

            bar = ax.bar(gi + offset, frac, width=bar_w, color=col,
                         alpha=0.85 if ti == 0 else 0.5,
                         edgecolor="white", linewidth=1.2, zorder=3)

            # Significance symbols only; effect-size/statistics described in caption
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            ax.text(gi + offset, frac + 0.010, sig,
                    ha="center", va="bottom", fontsize=12.5,
                    color="black", fontweight="bold")

    # Background dashed lines
    for tl in tail_labels:
        ax.axhline(bg[tl], color=TAIL_COLOR[tl], ls="--", lw=2.2, alpha=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(enrich_groups, fontsize=14.0)
    ax.set_ylabel("Fraction in tail", fontsize=15.5, fontweight="bold")
    ax.set_ylim(0, 0.40)

    patches = [mpatches.Patch(facecolor=TAIL_COLOR[tl], label=tl, edgecolor="white")
               for tl in tail_labels]
    leg = ax.legend(handles=patches, fontsize=12.5, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), ncol=2,
                    framealpha=0.0, edgecolor="none")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if show_title:
        ax.set_title("Tail enrichment in hematopoietic contexts", loc="left",
                     fontsize=15.0, fontweight="bold")
    if show_label:
        panel_label(ax, "(c)", size=16.0)


# ---------------------------------------------------------------------------
# Panel (d): top myeloid models
# ---------------------------------------------------------------------------
def panel_d_top_models(ax: plt.Axes, top_rows: list[dict], *, show_title: bool = True, show_label: bool = True) -> None:
    # Sort weakest→strongest (most negative at top)
    rows   = sorted(top_rows, key=lambda r: pf(r["ATP6V1B2"]))
    labels = [r["CellLineName"] for r in rows]
    vals   = [pf(r["ATP6V1B2"]) for r in rows]
    colors = [DISEASE_COLOR.get(r["DiseaseGroup"], PALETTE["neutral"]) for r in rows]

    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=1.2,
            height=0.62, zorder=3)

    # Value labels outside bar (right of left tip)
    for i, val in enumerate(vals):
        ax.text(val - 0.05, y[i], f"{val:.2f}",
                va="center", ha="right", fontsize=11.5,
                color="white", fontweight="bold", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13.5)
    ax.set_xlabel("ATP6V1B2 Chronos score", fontsize=15.5, fontweight="bold")
    ax.set_xlim(min(vals) - 0.35, 0)
    ax.axvline(-1.0, color=PALETTE["immune"], ls="--", lw=2.4, zorder=2)
    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(facecolor=DISEASE_COLOR[k], label=k, edgecolor="white")
               for k in DISEASE_COLOR]
    leg = ax.legend(handles=patches, fontsize=12.0, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), ncol=3,
                    framealpha=0.0, edgecolor="none")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if show_title:
        ax.set_title("Most dependent myeloid models", loc="left",
                     fontsize=15.0, fontweight="bold")
    if show_label:
        panel_label(ax, "(d)", size=16.0)


def polish_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=14.5, width=2.2, length=5.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)


def save_single_panel(path_base: Path, draw_fn, *args, figsize=(6.6, 5.0)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax, *args, show_title=False, show_label=False)
    polish_axis(ax)
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    scores   = read_csv(TABLES / "atp6v1b2_merged_scores.csv")
    tail_rows = read_csv(TABLES / "atp6v1b2_tail_enrichment.csv")
    top_rows  = read_csv(TABLES / "atp6v1b2_top_myeloid_models_refined.csv")

    fig = plt.figure(figsize=(14.0, 9.4))
    gs  = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.40,
                           left=0.09, right=0.97, top=0.95, bottom=0.11)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a_violin(ax_a, scores)
    panel_b_cdf(ax_b, scores)
    panel_c_enrichment(ax_c, tail_rows)
    panel_d_top_models(ax_d, top_rows)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        polish_axis(ax)

    plt.close(fig)
    NPJ_OUT.mkdir(parents=True, exist_ok=True)
    save_single_panel(NPJ_OUT / "fig_20a_depmap_violin", panel_a_violin, scores, figsize=(6.8, 5.2))
    save_single_panel(NPJ_OUT / "fig_20b_depmap_cdf", panel_b_cdf, scores, figsize=(6.8, 5.2))
    save_single_panel(NPJ_OUT / "fig_20c_depmap_tail_enrichment", panel_c_enrichment, tail_rows, figsize=(6.8, 5.2))
    save_single_panel(NPJ_OUT / "fig_20d_depmap_top_myeloid", panel_d_top_models, top_rows, figsize=(7.0, 5.4))
    print(f"Saved 20a-20d depmap panels to {NPJ_OUT}")


if __name__ == "__main__":
    main()
