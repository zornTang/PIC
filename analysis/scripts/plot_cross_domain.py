#!/usr/bin/env python3
"""
Generate cross-domain visualizations for the human-model transfer experiment.

Inputs:
  analysis/results/cross_domain/human_model_cross_domain_summary.csv

Outputs:
  analysis/results/cross_domain/cross_domain_auroc_heatmap.png
  analysis/results/cross_domain/cross_domain_auprc_heatmap.png
  analysis/results/cross_domain/cross_domain_test_delta_bar.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(summary: pd.DataFrame, metric: str, output_dir: Path) -> None:
    pivot = (
        summary.pivot(index="cell_line", columns="split", values=metric)
        .loc[sorted(summary["cell_line"].unique())]
    )
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(
        pivot.values,
        cmap="viridis",
        vmin=pivot.values.min(),
        vmax=pivot.values.max(),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric.upper()} across cell lines", fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                f"{pivot.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"cross_domain_{metric}_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline_test_auc = 0.9272102107150044
    baseline_test_auprc = 0.8244812201646308
    test_df = summary[summary["split"] == "test"].copy()
    test_df["delta_auc"] = test_df["auroc"] - baseline_test_auc
    test_df["delta_auprc"] = test_df["auprc"] - baseline_test_auprc
    test_df = test_df.sort_values("delta_auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(test_df))
    width = 0.35
    ax.bar(
        [xi - width / 2 for xi in x],
        test_df["delta_auc"],
        width=width,
        color="#1f78b4",
        label="ΔAUROC vs human",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        test_df["delta_auprc"],
        width=width,
        color="#e31a1c",
        label="ΔAUPRC vs human",
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(test_df["cell_line"], rotation=45, ha="right")
    ax.set_ylabel("Delta vs human baseline")
    ax.set_title("Human model cross-domain performance drop (test split)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_domain_test_delta_bar.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot cross-domain transfer results.")
    parser.add_argument(
        "--summary_csv",
        default="analysis/results/cross_domain/human_model_cross_domain_summary.csv",
        help="Input CSV containing columns: cell_line, split, auroc, auprc.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/cross_domain",
        help="Directory to store generated figures.",
    )
    args = parser.parse_args()
    summary = pd.read_csv(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(summary, "auroc", output_dir)
    plot_heatmap(summary, "auprc", output_dir)
    plot_delta_bar(summary, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
