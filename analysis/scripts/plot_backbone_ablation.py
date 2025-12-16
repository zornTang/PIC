#!/usr/bin/env python3
"""
Generate richer visualizations for backbone ablation runs.

Inputs:
  - CSV produced by analysis/scripts/ablation_evaluation.py

Outputs:
  - backbone_metrics.png : grouped AUROC/AUPRC bars per split
  - backbone_delta.png   : delta vs baseline bars
  - backbone_param_tradeoff.png : parameter-count vs performance scatter
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Color palette inspired by attention_head plots (blue/orange/green series)
PALETTE = {
    "blue": "#5DA5DA",
    "orange": "#FAA43A",
    "green": "#60BD68",
    "red": "#F17CB0",
    "purple": "#B2912F",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot backbone ablation summaries.")
    parser.add_argument(
        "--metrics_csv",
        default="analysis/results/cross_domain/structural_ablation_human/ablation_metrics.csv",
        help="CSV created by ablation_evaluation.py containing AUROC/AUPRC entries.",
    )
    parser.add_argument(
        "--baseline",
        default="attention",
        help="Experiment name treated as baseline for delta plots.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/cross_domain/structural_ablation_human",
        help="Directory to store generated figures.",
    )
    return parser.parse_args()


def load_metrics(csv_path: Path, baseline: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if baseline not in df["experiment"].unique():
        raise ValueError(f"Baseline '{baseline}' not found in {csv_path}")
    # Ensure consistent ordering
    df["experiment"] = df["experiment"].astype(str)
    df["split"] = df["split"].astype(str)
    df["variant_label"] = df["experiment"].str.capitalize()
    return df


def _prettify_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def plot_metric_bars(df: pd.DataFrame, output_dir: Path) -> None:
    splits = ["test"]
    metrics = ["auroc", "auprc"]
    variant_order: List[str] = sorted(df["experiment"].unique())
    label_map: Dict[str, str] = {name: name.capitalize() for name in variant_order}
    colors = {"test": PALETTE["orange"]}
    width = 0.35
    x = np.arange(len(variant_order))
    num_splits = len(splits)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, split in enumerate(splits):
            subset = (
                df[df["split"] == split]
                .set_index("experiment")
                .reindex(variant_order)
            )
            bars = ax.bar(
                x + (i - (num_splits - 1) / 2) * width,
                subset[metric],
                width=width,
                label=split.upper(),
                color=colors[split],
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.008,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#222222",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [label_map[v] for v in variant_order],
            fontweight="bold",
        )
        ax.set_ylabel(f"Test {metric.upper()}", fontweight="bold")
        ax.set_title(f"Test {metric.upper()} across backbones", fontweight="bold")
        ax.set_ylim(0.6, 1.02)
        _prettify_axes(ax)
        if len(splits) > 1:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(
            output_dir / f"backbone_metrics_{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_delta_bars(df: pd.DataFrame, baseline: str, output_path: Path) -> None:
    df = df.copy()
    splits = ["test"]
    metrics = ["delta_auroc_vs_baseline", "delta_auprc_vs_baseline"]
    metric_labels = {"delta_auroc_vs_baseline": "ΔAUROC vs baseline", "delta_auprc_vs_baseline": "ΔAUPRC vs baseline"}
    variant_order = [v for v in sorted(df["experiment"].unique()) if v != baseline]
    label_map = {name: name.capitalize() for name in variant_order}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 6), sharex=True)
    width = 0.35
    x = np.arange(len(variant_order))
    num_splits = len(splits)

    for ax, metric in zip(axes, metrics):
        for i, split in enumerate(splits):
            subset = (
                df[(df["split"] == split) & df["experiment"].isin(variant_order)]
                .set_index("experiment")
                .reindex(variant_order)
            )
            ax.bar(
                x + (i - (num_splits - 1) / 2) * width,
                subset[metric],
                width=width,
                label=split.upper() if metric == metrics[0] else None,
                color=PALETTE["green"],
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_ylabel(metric_labels[metric])
        _prettify_axes(ax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([label_map[v] for v in variant_order])
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Test delta vs attention baseline", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_param_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    test_df = df[df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        test_df["param_count"] / 1e6,
        test_df["auprc"],
        c=test_df["auroc"],
        cmap=plt.get_cmap("coolwarm"),
        s=150,
        edgecolor="black",
    )
    offsets: Dict[str, Tuple[int, int]] = {
        "attention": (0, 10),
        "cnn": (0, -15),
        "avgpool": (0, 10),
    }
    for _, row in test_df.iterrows():
        offset = offsets.get(row["experiment"], (0, 8))
        ax.annotate(
            row["experiment"].capitalize(),
            (row["param_count"] / 1e6, row["auprc"]),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Parameter efficiency - test split")
    x_vals = test_df["param_count"] / 1e6
    y_vals = test_df["auprc"]
    x_pad = max(0.05, (x_vals.max() - x_vals.min()) * 0.15)
    y_pad = max(0.02, (y_vals.max() - y_vals.min()) * 0.2)
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, min(1.0, y_vals.max() + y_pad))
    cbar = fig.colorbar(scatter)
    cbar.set_label("Test AUROC")
    ax.grid(alpha=0.3, linestyle="--")
    _prettify_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = Path(args.metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(csv_path, args.baseline)

    plot_metric_bars(df, output_dir)
    plot_delta_bars(df, args.baseline, output_dir / "backbone_delta.png")
    plot_param_tradeoff(df, output_dir / "backbone_param_tradeoff.png")
    print(f"Saved backbone plots to {output_dir}")


if __name__ == "__main__":
    main()
