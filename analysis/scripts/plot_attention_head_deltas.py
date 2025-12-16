#!/usr/bin/env python3
"""
Plot delta AUROC/AUPRC metrics for the attention-head ablation runs.

Inputs:
  analysis/results/ablations_heads/ablation_metrics.csv

Outputs:
  analysis/results/ablations_heads/ablation_metric_deltas_<split>.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attention head ablation deltas.")
    parser.add_argument(
        "--metrics_csv",
        default="analysis/results/ablations_heads/ablation_metrics.csv",
        help="CSV with columns experiment, split, delta_auroc_vs_baseline, delta_auprc_vs_baseline.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/ablations_heads",
        help="Directory where the PNG figures will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Dataset splits to plot.",
    )
    return parser.parse_args()


def _sort_key(experiment: str):
    if experiment.startswith("h") and experiment[1:].isdigit():
        return (0, int(experiment[1:]))
    return (1, experiment)


def plot_split(df: pd.DataFrame, split: str, output_dir: Path) -> None:
    subset = df[(df["split"] == split) & (df["experiment"] != "baseline")].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split '{split}'.")
    subset = subset.sort_values(by="experiment", key=lambda s: s.map(_sort_key))
    experiments = subset["experiment"].tolist()
    x = range(len(subset))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    color_auroc = "#1f78b4"
    color_auprc = "#33a02c"
    bars_auroc = ax.bar(
        [xi - width / 2 for xi in x],
        subset["delta_auroc_vs_baseline"],
        width=width,
        color=color_auroc,
        label="ΔAUROC vs baseline",
    )
    bars_auprc = ax.bar(
        [xi + width / 2 for xi in x],
        subset["delta_auprc_vs_baseline"],
        width=width,
        color=color_auprc,
        label="ΔAUPRC vs baseline",
    )
    ax.axhline(0, color="#444444", linewidth=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    y_values = pd.concat([subset["delta_auroc_vs_baseline"], subset["delta_auprc_vs_baseline"]])
    y_min, y_max = y_values.min(), y_values.max()
    span = y_max - y_min
    margin = 0.08 * span if span > 0 else 0.05
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments)
    ax.tick_params(axis="both", labelsize=11, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("Performance delta", fontweight="bold")
    ax.set_title(f"Attention head ablation deltas ({split} split)", fontweight="bold")
    legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), prop={"weight": "bold"})

    def annotate(bars):
        for bar in bars:
            height = bar.get_height()
            offset = 0.005 if height >= 0 else -0.005
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:+.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

    annotate(bars_auroc)
    annotate(bars_auprc)

    fig.subplots_adjust(bottom=0.24, right=0.78, top=0.92)
    fig.savefig(output_dir / f"ablation_metric_deltas_{split}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)
    for split in args.splits:
        plot_split(df, split, output_dir)
    print(f"Saved plots for splits: {', '.join(args.splits)}")


if __name__ == "__main__":
    main()
