#!/usr/bin/env python3
"""
Plot ROC/PR curves for attention-head ablations on the test split.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Palette sampled from attention_head_metrics figure (viridis-like gradient)
PALETTE = ["#482878", "#2D708E", "#1F9E89", "#6CCE59", "#FDE725"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot attention head ROC/PR curves.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "baseline=result/model_train_results/PIC_human",
            "h2=result/ablations/attn_heads2/PIC_human",
            "h4=result/ablations/attn_heads4/PIC_human",
            "h8=result/ablations/attn_heads8/PIC_human",
            "h16=result/ablations/attn_heads16/PIC_human",
        ],
        help="NAME=PATH entries pointing to run directories with *_pred_scores.npy and *_targets.npy.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split to use (default: test).",
    )
    parser.add_argument(
        "--roc_out",
        default="analysis/results/ablations_heads/attention_head_test_roc.png",
        help="Output path for ROC-only plot.",
    )
    parser.add_argument(
        "--pr_out",
        default="analysis/results/ablations_heads/attention_head_test_pr.png",
        help="Output path for PR-only plot.",
    )
    parser.add_argument(
        "--panel_out",
        default="analysis/results/ablations_heads/attention_head_test_roc_pr.png",
        help="Output path for combined ROC/PR figure.",
    )
    return parser.parse_args()


def load_npz(run_dir: Path, split: str):
    scores = np.load(run_dir / f"{split}_pred_scores.npy")
    targets = np.load(run_dir / f"{split}_targets.npy")
    return targets.reshape(-1), scores.reshape(-1)


def prepare_entries(args) -> List[Dict]:
    entries = []
    for entry in args.runs:
        if "=" not in entry:
            raise ValueError(f"Run entry '{entry}' must be NAME=PATH format.")
        name, path = entry.split("=", 1)
        run_dir = Path(path).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"{run_dir} not found.")
        y_true, y_score = load_npz(run_dir, args.split)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        entries.append(
            {
                "name": name,
                "run_dir": run_dir,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": auc(fpr, tpr),
                "recall": recall,
                "precision": precision,
                "pr_auc": average_precision_score(y_true, y_score),
                "pos_rate": float(np.mean(y_true)),
            }
        )
    return entries


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", labelsize=11, width=1.1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_curves(entries: List[Dict], kind: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for idx, item in enumerate(entries):
        color = PALETTE[idx % len(PALETTE)]
        is_baseline = item["name"].lower() in ("baseline", "h1", "head-1", "1h")
        label_name = "Baseline" if is_baseline else item["name"].upper()
        if kind == "roc":
            ax.plot(
                item["fpr"],
                item["tpr"],
                color=color,
                linewidth=2.3 if is_baseline else 2.0,
                linestyle="-",
                label=f"{label_name} (AUC={item['roc_auc']:.3f})",
            )
        else:
            ax.plot(
                item["recall"],
                item["precision"],
                color=color,
                linewidth=2.3 if is_baseline else 2.0,
                linestyle="-",
                label=f"{label_name} (AP={item['pr_auc']:.3f})",
            )

    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.0, label="Random chance")
        _style_axes(ax, "False Positive Rate", "True Positive Rate", "Test ROC curves by attention heads")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        pos_rate = entries[0]["pos_rate"]
        ax.plot([0, 1], [pos_rate] * 2, color="#9ca3af", linestyle="--", linewidth=1.0, alpha=0.4, label=f"Pos rate={pos_rate:.3f}")
        _style_axes(ax, "Recall", "Precision", "Test PR curves by attention heads")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left" if kind == "pr" else "lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel(entries: List[Dict], roc_out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for idx, item in enumerate(entries):
        color = PALETTE[idx % len(PALETTE)]
        is_baseline = item["name"].lower() in ("baseline", "h1", "head-1", "1h")
        linestyle = "-" if is_baseline else "--"
        label_name = "Baseline" if is_baseline else item["name"].upper()
        axes[0].plot(
            item["fpr"],
            item["tpr"],
            color=color,
            linewidth=2.3 if is_baseline else 2.0,
            linestyle=linestyle,
            label=f"{label_name} (AUC={item['roc_auc']:.3f})",
        )
        axes[1].plot(
            item["recall"],
            item["precision"],
            color=color,
            linewidth=2.3 if is_baseline else 2.0,
            linestyle=linestyle,
            label=f"{label_name} (AP={item['pr_auc']:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.0, label="Random chance")
    pos_rate = entries[0]["pos_rate"]
    axes[1].plot([0, 1], [pos_rate] * 2, color="#9ca3af", linestyle="--", linewidth=1.0, alpha=0.4, label=f"Pos rate={pos_rate:.3f}")
    _style_axes(axes[0], "False Positive Rate", "True Positive Rate", "Test ROC curves")
    _style_axes(axes[1], "Recall", "Precision", "Test PR curves")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(roc_out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    entries = prepare_entries(args)
    plot_curves(entries, "roc", Path(args.roc_out))
    plot_curves(entries, "pr", Path(args.pr_out))
    plot_panel(entries, Path(args.panel_out))
    print("Saved ROC/PR figures.")


if __name__ == "__main__":
    main()
