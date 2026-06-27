#!/usr/bin/env python3
"""
Figure 3.1: Human-level vs Immune-level model comparison — unified thesis style.

Five panels:
  (a) Scatter:          PES_human vs PES_immune, colored by subgroup
  (b) Threshold curves: proteins selected at each PES cutoff
  (c) KDE distribution: PES density for both models
  (d) ΔPES histogram:   score difference distribution
  (e) Bland–Altman:     agreement plot

Output: mythesis/figures/model_comparison.pdf / .png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from _fg_paths import FIGURES_NPJ_DIR, PIC2_PREDICTIONS_DIR
from style_config import apply_style, PALETTE, panel_label

apply_style()
# Dense data: keep all four spines; grid off (too noisy in scatter/histogram)
plt.rcParams["axes.spines.top"]   = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.grid"]         = False
# Base font slightly larger so ticks are legible after scaling to ~5.9-in linewidth
plt.rcParams["font.size"]         = 9
plt.rcParams["axes.labelsize"]    = 10
plt.rcParams["axes.titlesize"]    = 10
plt.rcParams["xtick.labelsize"]   = 8.5
plt.rcParams["ytick.labelsize"]   = 8.5
plt.rcParams["legend.fontsize"]   = 7.5
plt.rcParams["axes.linewidth"]    = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.major.size"]  = 3.5
plt.rcParams["ytick.major.size"]  = 3.5
plt.rcParams["lines.linewidth"]   = 1.6

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRED    = PIC2_PREDICTIONS_DIR
THESIS  = FIGURES_NPJ_DIR

IMMUNE_CSV = PRED / "neutrophil_immune_ensemble_predictions.csv"
HUMAN_CSV  = PRED / "neutrophil_proteins_human_predictions.csv"

THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# Colors — map to unified palette
# ---------------------------------------------------------------------------
COLORS = {
    "Immune-only High Essential": PALETTE["immune"],   # burnt orange
    "Human-only High Essential":  PALETTE["human"],    # steel blue
    "Both-High Essential":        PALETTE["common"],   # teal
    "Commonly Non-essential":     PALETTE["neutral"],  # light gray
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df_imm = pd.read_csv(IMMUNE_CSV)[["protein_id", "PES_score", "prediction"]]
    df_hum = pd.read_csv(HUMAN_CSV )[["protein_id", "PES_score", "prediction"]]
    df = pd.merge(df_imm, df_hum, on="protein_id",
                  suffixes=("_immune", "_human"))

    imm_hi = df["PES_score_immune"] >= THRESHOLD
    hum_hi = df["PES_score_human"]  >= THRESHOLD
    df["subgroup"] = "Commonly Non-essential"
    df.loc[ imm_hi &  hum_hi, "subgroup"] = "Both-High Essential"
    df.loc[~imm_hi &  hum_hi, "subgroup"] = "Human-only High Essential"
    df.loc[ imm_hi & ~hum_hi, "subgroup"] = "Immune-only High Essential"

    df["delta_PES"] = df["PES_score_immune"] - df["PES_score_human"]
    print("Subgroup counts:")
    print(df["subgroup"].value_counts())
    return df


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_a_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter: Human PES vs Immune PES, colored by subgroup."""
    order  = ["Commonly Non-essential", "Human-only High Essential",
              "Both-High Essential",    "Immune-only High Essential"]
    sizes  = {"Commonly Non-essential": 4, "Human-only High Essential": 8,
              "Both-High Essential": 9,    "Immune-only High Essential": 9}
    alphas = {"Commonly Non-essential": 0.20, "Human-only High Essential": 0.55,
              "Both-High Essential": 0.65,    "Immune-only High Essential": 0.70}

    for grp in order:
        sub = df[df["subgroup"] == grp]
        ax.scatter(sub["PES_score_human"], sub["PES_score_immune"],
                   c=COLORS[grp], s=sizes[grp], alpha=alphas[grp],
                   edgecolors="none",
                   zorder=3 if grp != "Commonly Non-essential" else 2,
                   rasterized=True)

    ax.axvline(THRESHOLD, color=PALETTE["ref_line"], ls="--", lw=1.0, zorder=4)
    ax.axhline(THRESHOLD, color=PALETTE["ref_line"], ls="--", lw=1.0, zorder=4)
    ax.fill_between([THRESHOLD, 1.02], THRESHOLD, 1.02,
                    alpha=0.06, color=COLORS["Both-High Essential"], zorder=1)
    ax.fill_between([0, THRESHOLD], THRESHOLD, 1.02,
                    alpha=0.06, color=COLORS["Immune-only High Essential"], zorder=1)
    ax.fill_between([THRESHOLD, 1.02], 0, THRESHOLD,
                    alpha=0.06, color=COLORS["Human-only High Essential"], zorder=1)

    r, p = stats.pearsonr(df["PES_score_human"], df["PES_score_immune"])
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.97, 0.04, f"r = {r:.3f},  {p_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE["neutral"], alpha=0.92))

    ax.set_xlim(0, 1.02);  ax.set_ylim(0, 1.02)
    ax.set_xlabel("Human-level PES")
    ax.set_ylabel("Immune-level PES")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    patches = [mpatches.Patch(facecolor=COLORS[g],
                               label=g.replace(" High Essential", "").replace("Commonly ", ""))
               for g in ["Immune-only High Essential", "Human-only High Essential",
                         "Both-High Essential",        "Commonly Non-essential"]]
    ax.legend(handles=patches, ncol=1,
              loc="upper left", bbox_to_anchor=(0.01, 0.98),
              framealpha=0.92, edgecolor=PALETTE["neutral"])
    panel_label(ax, "(a)", size=10)


def panel_b_threshold(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Threshold sensitivity curves."""
    ts    = np.linspace(0, 1, 300)
    n_hum = [(df["PES_score_human"]  >= t).sum() for t in ts]
    n_imm = [(df["PES_score_immune"] >= t).sum() for t in ts]

    ax.plot(ts, n_hum, color=PALETTE["human"],  label="Human-level",  zorder=3)
    ax.plot(ts, n_imm, color=PALETTE["immune"], label="Immune-level", zorder=3)
    ax.fill_between(ts, n_hum, n_imm,
                    where=[nh > ni for nh, ni in zip(n_hum, n_imm)],
                    alpha=0.14, color=PALETTE["human"], zorder=2)
    ax.fill_between(ts, n_hum, n_imm,
                    where=[ni >= nh for nh, ni in zip(n_hum, n_imm)],
                    alpha=0.14, color=PALETTE["immune"], zorder=2)
    ax.axvline(THRESHOLD, color=PALETTE["ref_line"], ls="--", lw=1.0,
               label=f"PES = {THRESHOLD}", zorder=4)

    ax.set_xlim(0, 1);  ax.set_ylim(0)
    ax.set_xlabel("PES threshold")
    ax.set_ylabel("Proteins retained")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(loc="upper right", framealpha=0.92, edgecolor=PALETTE["neutral"])
    panel_label(ax, "(b)", size=10)


def panel_c_kde(ax: plt.Axes, df: pd.DataFrame) -> None:
    """KDE density of PES scores."""
    for col, color, label in [
        ("PES_score_human",  PALETTE["human"],  "Human-level"),
        ("PES_score_immune", PALETTE["immune"], "Immune-level"),
    ]:
        xs  = np.linspace(0, 1, 400)
        kde = gaussian_kde(df[col].values, bw_method=0.12)
        ys  = kde(xs)
        ax.plot(xs, ys, color=color, label=label, zorder=3)
        ax.fill_between(xs, ys, alpha=0.12, color=color, zorder=2)

    ax.axvline(THRESHOLD, color=PALETTE["ref_line"], ls="--", lw=1.0,
               label=f"PES = {THRESHOLD}")
    ax.set_xlim(0, 1);  ax.set_ylim(0)
    ax.set_xlabel("PES score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.92, edgecolor=PALETTE["neutral"])
    panel_label(ax, "(c)", size=10)


def panel_d_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    """ΔPES histogram."""
    delta = df["delta_PES"].values
    mn    = delta.mean()

    ax.hist(delta, bins=35, color=PALETTE["neutral"],
            edgecolor="white", linewidth=0.3, zorder=2)
    pos, neg = delta[delta >= 0], delta[delta < 0]
    if len(pos):
        ax.hist(pos, bins=25, color=PALETTE["immune"], alpha=0.65,
                edgecolor="white", linewidth=0.3, zorder=3, label="Immune > Human")
    if len(neg):
        ax.hist(neg, bins=25, color=PALETTE["human"], alpha=0.65,
                edgecolor="white", linewidth=0.3, zorder=3, label="Human > Immune")

    ax.axvline(0,  color=PALETTE["ref_line"], ls="--", lw=1.0, zorder=4)
    ax.axvline(mn, color="#555", ls=":", lw=1.0, zorder=4,
               label=f"Mean = {mn:.3f}")

    ax.set_xlabel("ΔPES  (Immune − Human)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor=PALETTE["neutral"])
    panel_label(ax, "(d)", size=10)


def panel_e_ba(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Bland–Altman agreement plot."""
    mean_pes = (df["PES_score_immune"] + df["PES_score_human"]) / 2
    diff_pes = df["PES_score_immune"] - df["PES_score_human"]
    md       = diff_pes.mean()
    sd       = diff_pes.std()
    loa_up, loa_lo = md + 1.96 * sd, md - 1.96 * sd

    colors = df["subgroup"].map(COLORS).values
    ax.scatter(mean_pes, diff_pes, c=colors,
               alpha=0.30, s=4, edgecolors="none",
               rasterized=True, zorder=2)

    ax.axhline(md,     color="#333", lw=1.4, ls="-",  zorder=4)
    ax.axhline(loa_up, color=PALETTE["immune"], lw=1.0, ls="--", zorder=4)
    ax.axhline(loa_lo, color=PALETTE["human"],  lw=1.0, ls="--", zorder=4)
    ax.axhline(0, color=PALETTE["ref_line"], lw=0.7, ls=":", zorder=3)
    ax.fill_between([0, 1], loa_lo, loa_up,
                    alpha=0.05, color=PALETTE["neutral"], zorder=1)

    # Direct labels on lines — avoid legend clutter
    ax.text(1.01, md,     f"{md:.3f}",    va="center", fontsize=6,
            color="#333", transform=ax.get_yaxis_transform())
    ax.text(1.01, loa_up, f"+{loa_up:.2f}", va="center", fontsize=6,
            color=PALETTE["immune"], transform=ax.get_yaxis_transform())
    ax.text(1.01, loa_lo, f"{loa_lo:.2f}", va="center", fontsize=6,
            color=PALETTE["human"],  transform=ax.get_yaxis_transform())

    pct = (((diff_pes >= loa_lo) & (diff_pes <= loa_up)).sum() / len(diff_pes) * 100)
    ax.text(0.04, 0.04, f"{pct:.1f}% within ±1.96 SD",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE["neutral"], alpha=0.9))

    ax.set_xlim(-0.01, 1.0)
    ax.set_xlabel("Mean PES")
    ax.set_ylabel("ΔPES  (Imm. − Hum.)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    panel_label(ax, "(e)", size=10)


def strip_single_panel(ax: plt.Axes) -> None:
    ax.set_title("")
    for text in list(ax.texts):
        if text.get_text() in {"(a)", "(b)", "(c)", "(d)", "(e)"}:
            text.remove()


def emphasize_single_panel(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=13, width=2.2, length=6.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontsize(16)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.3)
    for line in ax.lines:
        line.set_linewidth(max(line.get_linewidth(), 3.0))
        if line.get_marker() not in (None, "None", ""):
            line.set_markersize(max(line.get_markersize(), 4.6))
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(11.5)
            text.set_fontweight("bold")


def export_legend_only(handles, labels, output_path: Path, ncol: int = 3) -> None:
    fig = plt.figure(figsize=(max(3.2, 1.25 * len(labels)), 0.9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_fontsize(11.5)
        text.set_fontweight("bold")
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


def export_single_panels(df: pd.DataFrame) -> None:
    outputs = [
        ("fig_12a_model_comparison_scatter", panel_a_scatter, (7.2, 4.6)),
        ("fig_12b_model_comparison_threshold", panel_b_threshold, (4.6, 4.6)),
        ("fig_12c_model_comparison_kde", panel_c_kde, (3.5, 4.2)),
        ("fig_12d_model_comparison_delta_hist", panel_d_delta, (3.5, 4.2)),
        ("fig_12e_model_comparison_bland_altman", panel_e_ba, (3.5, 4.2)),
    ]
    for stem, panel_fn, figsize in outputs:
        fig, ax = plt.subplots(figsize=figsize)
        panel_fn(ax, df)
        strip_single_panel(ax)
        emphasize_single_panel(ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        fig.savefig(THESIS / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
        plt.close(fig)

    subgroup_handles = [
        mpatches.Patch(facecolor=COLORS["Immune-only High Essential"], label="Immune-only"),
        mpatches.Patch(facecolor=COLORS["Human-only High Essential"], label="Human-only"),
        mpatches.Patch(facecolor=COLORS["Both-High Essential"], label="Both-high"),
        mpatches.Patch(facecolor=COLORS["Commonly Non-essential"], label="Non-essential"),
    ]
    export_legend_only(
        subgroup_handles,
        [h.get_label() for h in subgroup_handles],
        THESIS / "fig_12_legend_subgroup.png",
        ncol=2,
    )

    model_handles = [
        Line2D([0], [0], color=PALETTE["human"], linewidth=2.4, label="Human-level"),
        Line2D([0], [0], color=PALETTE["immune"], linewidth=2.4, label="Immune-level"),
    ]
    export_legend_only(
        model_handles,
        [h.get_label() for h in model_handles],
        THESIS / "fig_12_legend_model.png",
        ncol=2,
    )

    delta_handles = [
        mpatches.Patch(facecolor=PALETTE["immune"], label="Immune > Human"),
        mpatches.Patch(facecolor=PALETTE["human"], label="Human > Immune"),
        Line2D([0], [0], color="#555", linestyle=":", linewidth=1.2, label="Mean"),
    ]
    export_legend_only(
        delta_handles,
        [h.get_label() for h in delta_handles],
        THESIS / "fig_12_legend_delta.png",
        ncol=3,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    export_single_panels(df)
    print(f"Saved 12a-12e model comparison panels to {THESIS}")


if __name__ == "__main__":
    main()
