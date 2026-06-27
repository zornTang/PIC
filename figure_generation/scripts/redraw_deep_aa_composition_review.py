#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from _fg_paths import FIGURES_DIR, PIC2_ANALYSIS_RESULTS, PIC2_PREDICTIONS_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_config import PALETTE, apply_style  # type: ignore
from visualize_deep_immune_analysis import DeepVisualizer  # type: ignore


OUT_DIR = FIGURES_DIR
OUT_STEM = OUT_DIR / "fig_aux_aa_composition_preview"

AA_PROPS = {
    "A": "Hydrophobic",
    "I": "Hydrophobic",
    "L": "Hydrophobic",
    "M": "Hydrophobic",
    "F": "Hydrophobic",
    "W": "Hydrophobic",
    "V": "Hydrophobic",
    "N": "Polar",
    "C": "Polar",
    "Q": "Polar",
    "S": "Polar",
    "T": "Polar",
    "Y": "Polar",
    "D": "Negative",
    "E": "Negative",
    "R": "Positive",
    "H": "Positive",
    "K": "Positive",
    "G": "Special",
    "P": "Special",
}

PROP_COLORS = {
    "Hydrophobic": PALETTE["common"],
    "Polar": PALETTE["neutral"],
    "Negative": PALETTE["immune"],
    "Positive": PALETTE["human"],
    "Special": "#B7791F",
}

REF_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


def build_heat_data() -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    base_dir = PIC2_PREDICTIONS_DIR
    analysis_dir = PIC2_ANALYSIS_RESULTS
    viz = DeepVisualizer(
        str(base_dir / "neutrophil_immune_ensemble_predictions.csv"),
        str(base_dir / "neutrophil_proteins_human_predictions.csv"),
        str(analysis_dir / "domain_enrichment_results.csv"),
        str(analysis_dir / "keyword_enrichment_results.csv"),
        str(analysis_dir / "subcellular_localization_distribution.csv"),
        str(OUT_DIR),
    )

    target_groups = ["Human-only High Essential", "Immune-only High Essential"]
    df_target = viz.df[viz.df["subgroup"].isin(target_groups)].copy()

    aa_freqs = []
    for _, row in df_target.iterrows():
        freqs = viz.calc_aa_freq(row["sequence"]) if hasattr(viz, "calc_aa_freq") else None
        if freqs is None:
            from visualize_deep_immune_analysis import calc_aa_freq  # type: ignore

            freqs = calc_aa_freq(row["sequence"])
        freqs["subgroup"] = row["subgroup"]
        aa_freqs.append(freqs)

    aa_df = pd.DataFrame(aa_freqs)
    aa_mean = aa_df.groupby("subgroup").mean() * 100.0
    heat_data = pd.DataFrame(
        {
            "Human Model": aa_mean.loc["Human-only High Essential"],
            "Immune Model": aa_mean.loc["Immune-only High Essential"],
        }
    ).T
    heat_data = heat_data[[aa for aa in REF_ORDER if aa in heat_data.columns]]
    delta = heat_data.loc["Immune Model"] - heat_data.loc["Human Model"]
    delta = delta[[aa for aa in REF_ORDER if aa in delta.index]]

    aa_pvals: dict[str, float] = {}
    for aa in REF_ORDER:
        human_aa = aa_df[aa_df["subgroup"] == "Human-only High Essential"][aa]
        immune_aa = aa_df[aa_df["subgroup"] == "Immune-only High Essential"][aa]
        if len(human_aa) > 0 and len(immune_aa) > 0:
            _, p = stats.mannwhitneyu(human_aa, immune_aa, alternative="two-sided")
            aa_pvals[aa] = p
        else:
            aa_pvals[aa] = 1.0
    return heat_data, delta, aa_pvals


def main() -> None:
    apply_style()
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.grid"] = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heat_data, delta, aa_pvals = build_heat_data()

    fig, ax = plt.subplots(figsize=(16, 7.4))
    sns.heatmap(
        heat_data,
        cmap="Greys",
        annot=False,
        vmin=0,
        vmax=10,
        cbar_kws={"label": "Percentage (%)", "ticks": [0, 10]},
        ax=ax,
        linewidths=0.6,
        linecolor="white",
    )

    abs_deltas = delta.abs().sort_values(ascending=False)
    top_divergent = set(abs_deltas.head(3).index)
    for y, row_name in enumerate(heat_data.index):
        for x, col_name in enumerate(heat_data.columns):
            val = heat_data.iloc[y, x]
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=11.5,
                color="black",
                alpha=0.92,
                fontweight="bold",
            )
            if y == 1:
                d_val = delta[col_name]
                p_val = aa_pvals.get(col_name, 1.0)
                if p_val < 0.05 or col_name in top_divergent:
                    ax.add_patch(
                        Rectangle((x, 0), 1, 2, fill=False, edgecolor=PALETTE["immune"], linewidth=1.8, zorder=5)
                    )
                    sign = "+" if d_val > 0 else "-"
                    sig_star = "*" if p_val < 0.05 else ""
                    ax.text(
                        x + 0.5,
                        y + 0.16,
                        f"Δ{sign}{abs(d_val):.1f}{sig_star}",
                        ha="center",
                        va="center",
                        fontsize=10.5,
                        color=PALETTE["immune"],
                        fontweight="bold",
                    )

    ax.set_title("")
    ax.set_ylabel("Model Exclusivity (PES ≥ 0.8)", fontsize=15, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=14, labelbottom=True, bottom=True, pad=10, width=1.8, length=5)
    ax.tick_params(axis="y", labelsize=14, left=True, width=1.8, length=5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    for x, col_name in enumerate(heat_data.columns):
        prop = AA_PROPS.get(col_name, "Special")
        ax.add_patch(Rectangle((x, 2.25), 1, 0.11, fill=True, color=PROP_COLORS[prop], clip_on=False))

    legend_elements = [Patch(facecolor=color, label=prop) for prop, color in PROP_COLORS.items()]
    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.02),
        framealpha=0.0,
        edgecolor="none",
        title="Amino Acid Property Classes",
        title_fontsize=14,
    )
    plt.setp(legend.get_title(), fontweight="bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.subplots_adjust(bottom=0.29, top=0.97)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_color("black")
    ax.axhline(0, color="black", linewidth=1.4)
    ax.axhline(heat_data.shape[0], color="black", linewidth=1.4)
    ax.axvline(0, color="black", linewidth=1.4)
    ax.axvline(heat_data.shape[1], color="black", linewidth=1.4)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13, width=1.6, length=4)
    cbar.set_label("Percentage (%)", fontsize=14, fontweight="bold")
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
