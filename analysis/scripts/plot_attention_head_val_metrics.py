#!/usr/bin/env python3
"""
Draw validation AUROC/AUPRC epoch curves for each attention-head variant.

Inputs:
  - analysis/results/ablations_heads/attention_head_summary.csv
  - result/model_train_results/PIC_human/PIC_human_val_result.csv
  - result/ablations/attn_heads{N}/PIC_human/PIC_human_val_result.csv

Outputs:
  - analysis/results/ablations_heads/attention_head_val_auc.png
  - analysis/results/ablations_heads/attention_head_val_auprc.png
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#472f7d", "#355c8c", "#24868e", "#35b579", "#ecec3c"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot validation epoch curves per attention head.")
    parser.add_argument(
        "--summary_csv",
        default="analysis/results/ablations_heads/attention_head_summary.csv",
        help="CSV describing each head variant (must include columns short and num_heads).",
    )
    parser.add_argument(
        "--dataset",
        default="PIC_human",
        help="Dataset name used in run directories.",
    )
    parser.add_argument(
        "--prefix",
        default="human",
        help="Column prefix in *_val_result.csv (e.g., human_epoch, human_val_auc).",
    )
    parser.add_argument(
        "--baseline_dir",
        default="result/model_train_results",
        help="Directory containing the baseline run subfolder.",
    )
    parser.add_argument(
        "--ablations_dir",
        default="result/ablations",
        help="Root directory containing attn_heads{N} runs.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/ablations_heads",
        help="Directory to write PNGs.",
    )
    return parser.parse_args()


def resolve_csv_path(num_heads: int, dataset: str, baseline_dir: Path, ablations_dir: Path) -> Path:
    if num_heads == 1:
        return baseline_dir / dataset / f"{dataset}_val_result.csv"
    return ablations_dir / f"attn_heads{num_heads}" / dataset / f"{dataset}_val_result.csv"


def load_runs(summary: pd.DataFrame, dataset: str, prefix: str, baseline_dir: Path, ablations_dir: Path) -> List[Dict]:
    runs: List[Dict] = []
    for _, row in summary.iterrows():
        num_heads = int(row["num_heads"])
        csv_path = resolve_csv_path(num_heads, dataset, baseline_dir, ablations_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for {row['short']} at {csv_path}")
        df = pd.read_csv(csv_path)
        epoch_col = f"{prefix}_epoch"
        auc_col = f"{prefix}_val_auc"
        pr_col = f"{prefix}_val_pr_auc"
        for col in (epoch_col, auc_col, pr_col):
            if col not in df.columns:
                raise KeyError(f"{col} missing in {csv_path}")
        runs.append(
            {
                "short": str(row["short"]),
                "label": str(row.get("label", row["short"])),
                "num_heads": num_heads,
                "epochs": df[epoch_col].to_numpy(),
                "val_auc": df[auc_col].to_numpy(),
                "val_pr_auc": df[pr_col].to_numpy(),
            }
        )
    return runs


def plot_epoch_curves(runs: List[Dict], metric: str, ylabel: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for idx, run in enumerate(runs):
        color = PALETTE[idx % len(PALETTE)]
        epochs = run["epochs"]
        values = run[metric]
        linestyle = "-" if run["num_heads"] == 1 else "--"
        linewidth = 2.6 if run["num_heads"] == 1 else 2.0
        marker = "o"
        markersize = 4.5 if run["num_heads"] == 1 else 4
        ax.plot(
            epochs,
            values,
            label=("Baseline" if run["num_heads"] == 1 else f"{run['short']} ({run['num_heads']} heads)"),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
        )

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(f"{ylabel} across epochs", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    y_vals = np.concatenate([run[metric] for run in runs])
    finite_vals = y_vals[np.isfinite(y_vals)]
    if finite_vals.size > 0:
        span = finite_vals.max() - finite_vals.min()
        pad = max(span * 0.08, 0.01)
        ax.set_ylim(finite_vals.min() - pad, finite_vals.max() + pad)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    fig.subplots_adjust(right=0.78, bottom=0.15, top=0.9)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    summary = summary.sort_values("num_heads")
    runs = load_runs(
        summary,
        dataset=args.dataset,
        prefix=args.prefix,
        baseline_dir=Path(args.baseline_dir),
        ablations_dir=Path(args.ablations_dir),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_epoch_curves(runs, "val_auc", "Validation AUROC", output_dir / "attention_head_val_auc.png")
    plot_epoch_curves(runs, "val_pr_auc", "Validation AUPRC", output_dir / "attention_head_val_auprc.png")
    print("Saved epoch-based validation plots.")


if __name__ == "__main__":
    main()
