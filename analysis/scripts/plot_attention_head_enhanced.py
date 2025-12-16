#!/usr/bin/env python3
"""
Extra visualizations for attention-head ablation runs.

Inputs:
  analysis/results/ablations_heads/ablation_metrics.csv

Outputs (default):
  analysis/results/ablations_heads/attention_head_delta_lines.png
  analysis/results/ablations_heads/attention_head_volcano_test.png
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced plots for attention-head ablation.")
    parser.add_argument(
        "--metrics_csv",
        default="analysis/results/ablations_heads/ablation_metrics.csv",
        help="CSV produced by ablation_evaluation.py for head ablations.",
    )
    parser.add_argument(
        "--baseline",
        default="baseline",
        help="Experiment name treated as baseline.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/ablations_heads",
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def _sort_key(name: str):
    if name.startswith("h") and name[1:].isdigit():
        return (0, int(name[1:]))
    if name == "baseline":
        return (-1, -1)
    return (1, name)


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["experiment"] = df["experiment"].astype(str)
    df["split"] = df["split"].astype(str)
    df = df.sort_values(by="experiment", key=lambda s: s.map(_sort_key))
    return df


def plot_delta_lines(df: pd.DataFrame, baseline: str, output_path: Path) -> None:
    # Work on copy to avoid modifying the original frame
    df = df.copy()
    # Replace NaN deltas (baseline rows) with 0 to draw reference line
    for col in ["delta_auroc_vs_baseline", "delta_auprc_vs_baseline"]:
        df[col] = df[col].fillna(0.0)

    experiments: List[str] = list(dict.fromkeys(df["experiment"].tolist()))
    label_map: Dict[str, str] = {
        "baseline": "Baseline",
        **{name: name.upper() if name.startswith("h") else name for name in experiments},
    }
    x = np.arange(len(experiments))
    metrics = [
        ("delta_auroc_vs_baseline", "ΔAUROC vs baseline"),
        ("delta_auprc_vs_baseline", "ΔAUPRC vs baseline"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    colors = {"val": "#5DA5DA", "test": "#F17CB0"}

    for ax, (metric_col, title) in zip(axes, metrics):
        for split in ["val", "test"]:
            subset = df[df["split"] == split].set_index("experiment").reindex(experiments)
            ax.plot(
                x,
                subset[metric_col],
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=7,
                color=colors[split],
                label=split.upper(),
            )
        ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--")
        ax.set_ylabel(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([label_map.get(name, name) for name in experiments], fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Attention head ablation deltas", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(df: pd.DataFrame, output_path: Path) -> None:
    """Volcano-style plot for test split: ΔAUPRC vs -log10(p)."""
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[test_df["experiment"] != "baseline"]
    if test_df.empty:
        raise ValueError("No test split rows found for volcano plot.")

    test_df["neg_log10_p"] = -np.log10(test_df["pvalue_auprc"])
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    scatter = ax.scatter(
        test_df["delta_auprc_vs_baseline"],
        test_df["neg_log10_p"],
        s=90,
        color="#60BD68",
        edgecolor="#1f2937",
        alpha=0.9,
    )
    for _, row in test_df.iterrows():
        ax.annotate(
            row["experiment"].upper(),
            (row["delta_auprc_vs_baseline"], row["neg_log10_p"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.axhline(-np.log10(0.05), color="#9ca3af", linestyle=":", linewidth=1.0, label="p=0.05")
    ax.set_xlabel("ΔAUPRC vs baseline", fontweight="bold")
    ax.set_ylabel("-log10(p-value)", fontweight="bold")
    ax.set_title("Test split: effect size vs significance", fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(Path(args.metrics_csv))
    plot_delta_lines(df, args.baseline, output_dir / "attention_head_delta_lines.png")
    plot_volcano(df, output_dir / "attention_head_volcano_test.png")
    print(f"Saved attention-head plots to {output_dir}")


if __name__ == "__main__":
    main()
