#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from _fg_paths import FIGURES_NPJ_DIR, PIC2_ABLATIONS_DIR, PIC2_MODEL_DIR
from style_config import apply_style, PALETTE, FIGSIZE, panel_label as _panel_label

PIC_HUMAN_DIR = PIC2_MODEL_DIR
ATTENTION_OUTPUT_DIR = FIGURES_NPJ_DIR
THESIS_FIGURES_DIR = FIGURES_NPJ_DIR
ATTENTION_RUNS = [
    ("baseline", "Baseline", 1, PIC2_MODEL_DIR),
    ("h2", "h2 (2 heads)", 2, PIC2_ABLATIONS_DIR / "attn_heads2" / "PIC_human"),
    ("h4", "h4 (4 heads)", 4, PIC2_ABLATIONS_DIR / "attn_heads4" / "PIC_human"),
    ("h8", "h8 (8 heads)", 8, PIC2_ABLATIONS_DIR / "attn_heads8" / "PIC_human"),
    ("h16", "h16 (16 heads)", 16, PIC2_ABLATIONS_DIR / "attn_heads16" / "PIC_human"),
]

# Performance curve colors from unified palette
PURPLE = PALETTE["roc"]
GREEN  = PALETTE["pr"]
ATTENTION_PALETTE = PALETTE["abl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate PIC summary figures from source artifacts.")
    parser.add_argument(
        "--figure",
        choices=["all", "pic_human", "attention_heads"],
        default="all",
        help="Which figure(s) to generate.",
    )
    parser.add_argument(
        "--panel-label-size",
        type=float,
        default=22.0,
        help="Font size for panel labels like (a), (b), (c), (d).",
    )
    return parser.parse_args()


def configure_style() -> None:
    apply_style()


def infer_prefix(columns: Iterable[str]) -> str:
    for column in columns:
        if column.endswith("_epoch"):
            return column[: -len("_epoch")]
    raise ValueError("Could not infer metric prefix from CSV columns.")


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(axis="both", width=1.6, length=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.9)


def add_panel_label(ax, label: str, size: float):
    return _panel_label(ax, label, size=size)


def make_single_panel(figsize: tuple[float, float] = (6.2, 4.8)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.grid(False)
    return fig, ax


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
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2.3)


def apply_perf_single_panel_fonts_08(ax: plt.Axes) -> None:
    """Larger explicit font scale for the 08 single-panel exports."""
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=13.5, width=2.2, length=6.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontsize(16)
    ax.title.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12)
            text.set_fontweight("bold")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2.3)


def apply_perf_single_panel_fonts_11(ax: plt.Axes) -> None:
    """Original explicit font scale for the 11 single-panel exports."""
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=12, width=2.2, length=6.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontsize(16)
    ax.title.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10.5)
            text.set_fontweight("bold")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2.3)


def export_legend_only(handles, labels, output_path: Path, ncol: int = 3) -> None:
    fig = plt.figure(figsize=(max(3.6, 1.25 * len(labels)), 0.9))
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


def plot_pic_human_summary(panel_label_size: float) -> None:
    val_csv = PIC_HUMAN_DIR / "PIC_human_val_result.csv"
    test_scores_path = PIC_HUMAN_DIR / "test_pred_scores.npy"
    test_targets_path = PIC_HUMAN_DIR / "test_targets.npy"

    df = pd.read_csv(val_csv)
    prefix = infer_prefix(df.columns)
    col = lambda name: f"{prefix}_{name}"

    epochs = df[col("epoch")].to_numpy()
    train_loss = df[col("train_loss")].to_numpy()
    val_loss = df[col("val_loss")].to_numpy()
    val_auc = df[col("val_auc")].to_numpy()
    val_pr_auc = df[col("val_pr_auc")].to_numpy()

    y_true = np.load(test_targets_path)
    y_score = np.load(test_scores_path)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['quad'])

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color=PURPLE, linewidth=2.3, marker="o", markersize=2.8, label="Train Loss")
    ax.plot(epochs, val_loss, color=GREEN, linewidth=2.3, marker="o", markersize=2.8, label="Val Loss")
    style_axes(ax, "Epoch", "Loss", "Loss Progression")
    ax.legend(frameon=False, loc="upper right")
    label_a = add_panel_label(ax, "(a)", panel_label_size)

    ax = axes[0, 1]
    ax.plot(epochs, val_auc, color=PURPLE, linewidth=2.3, marker="o", markersize=2.8, label="Val AUC")
    ax.plot(epochs, val_pr_auc, color=GREEN, linewidth=2.3, marker="o", markersize=2.8, label="Val PR AUC")
    style_axes(ax, "Epoch", "Score", "Metric Progression")
    metric_min = min(val_auc.min(), val_pr_auc.min())
    metric_max = max(val_auc.max(), val_pr_auc.max())
    metric_pad = max((metric_max - metric_min) * 0.12, 0.005)
    ax.set_ylim(metric_min - metric_pad, metric_max + metric_pad * 0.4)
    ax.legend(frameon=False, loc="lower right")
    label_b = add_panel_label(ax, "(b)", panel_label_size)

    ax = axes[1, 0]
    ax.plot(fpr, tpr, color=PURPLE, linewidth=2.7, label=f"AUC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.3)
    style_axes(ax, "False Positive Rate", "True Positive Rate", "ROC Curve")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(frameon=False, loc="lower right")
    label_c = add_panel_label(ax, "(c)", panel_label_size)

    ax = axes[1, 1]
    ax.plot(recall, precision, color=GREEN, linewidth=2.7, label=f"PR AUC = {auprc:.3f}")
    style_axes(ax, "Recall", "Precision", "PR Curve")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.legend(frameon=False, loc="lower left")
    label_d = add_panel_label(ax, "(d)", panel_label_size)

    plt.close(fig)

    single_outputs = [
        ("fig_08a_training_loss.png", "loss"),
        ("fig_08b_validation_metrics.png", "metrics"),
        ("fig_08c_test_roc.png", "roc"),
        ("fig_08d_test_pr.png", "pr"),
    ]
    for filename, panel in single_outputs:
        single_fig, ax = make_single_panel()
        if panel == "loss":
            ax.plot(epochs, train_loss, color=PURPLE, linewidth=3.4, marker="o", markersize=4.4, label="Train Loss")
            ax.plot(epochs, val_loss, color=GREEN, linewidth=3.4, marker="o", markersize=4.4, label="Val Loss")
            style_axes(ax, "Epoch", "Loss", "Loss Progression")
            ax.set_title("")
            ax.legend(frameon=False, loc="upper right")
        elif panel == "metrics":
            ax.plot(epochs, val_auc, color=PURPLE, linewidth=3.4, marker="o", markersize=4.4, label="Val AUC")
            ax.plot(epochs, val_pr_auc, color=GREEN, linewidth=3.4, marker="o", markersize=4.4, label="Val PR AUC")
            style_axes(ax, "Epoch", "Score", "Metric Progression")
            ax.set_title("")
            ax.set_ylim(metric_min - metric_pad, metric_max + metric_pad * 0.4)
            ax.legend(frameon=False, loc="lower right")
        elif panel == "roc":
            ax.plot(fpr, tpr, color=PURPLE, linewidth=3.5, label=f"AUC = {auroc:.3f}")
            ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.8)
            style_axes(ax, "False Positive Rate", "True Positive Rate", "ROC Curve")
            ax.set_title("")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.legend(frameon=False, loc="lower right")
        else:
            ax.plot(recall, precision, color=GREEN, linewidth=3.5, label=f"PR AUC = {auprc:.3f}")
            style_axes(ax, "Recall", "Precision", "PR Curve")
            ax.set_title("")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(0.0, 1.01)
            ax.legend(frameon=False, loc="lower left")
        apply_perf_single_panel_fonts_08(ax)
        single_fig.tight_layout()
        single_fig.savefig(THESIS_FIGURES_DIR / filename, dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
        plt.close(single_fig)


def load_attention_run(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    val_csv = run_dir / "PIC_human_val_result.csv"
    test_targets = run_dir / "test_targets.npy"
    test_scores = run_dir / "test_pred_scores.npy"
    if not (val_csv.exists() and test_targets.exists() and test_scores.exists()):
        raise FileNotFoundError(f"Missing required attention-head inputs under {run_dir}")
    return pd.read_csv(val_csv), np.load(test_targets), np.load(test_scores)


def plot_attention_heads_summary(panel_label_size: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE['quad'])
    auprc_series = []
    auroc_series = []
    roc_entries = []
    pr_entries = []

    for color, (short, label, _, run_dir) in zip(ATTENTION_PALETTE, ATTENTION_RUNS):
        df, y_true, y_score = load_attention_run(run_dir)
        prefix = infer_prefix(df.columns)
        auprc_series.append(
            {
                "label": label,
                "short": short,
                "epochs": df[f"{prefix}_epoch"].to_numpy(),
                "values": df[f"{prefix}_val_pr_auc"].to_numpy(),
                "color": color,
            }
        )
        auroc_series.append(
            {
                "label": label,
                "short": short,
                "epochs": df[f"{prefix}_epoch"].to_numpy(),
                "values": df[f"{prefix}_val_auc"].to_numpy(),
                "color": color,
            }
        )
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_entries.append(
            {
                "label": "Baseline" if short == "baseline" else short.upper(),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(y_true, y_score),
                "color": color,
                "short": short,
            }
        )
        pr_entries.append(
            {
                "label": "Baseline" if short == "baseline" else short.upper(),
                "recall": recall,
                "precision": precision,
                "ap": average_precision_score(y_true, y_score),
                "color": color,
                "short": short,
                "pos_rate": float(np.mean(y_true)),
            }
        )

    ax = axes[0, 0]
    for item in auprc_series:
        is_baseline = item["short"] == "baseline"
        ax.plot(
            item["epochs"],
            item["values"],
            color=item["color"],
            linewidth=2.4 if is_baseline else 2.1,
            linestyle="-" if is_baseline else "--",
            marker="o",
            markersize=3.2,
            label=item["label"],
        )
    style_axes(ax, "Epoch", "Validation AUPRC", "Validation AUPRC across epochs")
    top_vals = np.concatenate([item["values"] for item in auprc_series])
    pad = max((top_vals.max() - top_vals.min()) * 0.12, 0.01)
    ax.set_ylim(top_vals.min() - pad, top_vals.max() + pad * 0.35)
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "(a)", panel_label_size)

    ax = axes[0, 1]
    for item in auroc_series:
        is_baseline = item["short"] == "baseline"
        ax.plot(
            item["epochs"],
            item["values"],
            color=item["color"],
            linewidth=2.4 if is_baseline else 2.1,
            linestyle="-" if is_baseline else "--",
            marker="o",
            markersize=3.2,
            label=item["label"],
        )
    style_axes(ax, "Epoch", "Validation AUROC", "Validation AUROC across epochs")
    top_vals = np.concatenate([item["values"] for item in auroc_series])
    pad = max((top_vals.max() - top_vals.min()) * 0.12, 0.01)
    ax.set_ylim(top_vals.min() - pad, top_vals.max() + pad * 0.35)
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "(b)", panel_label_size)

    ax = axes[1, 0]
    for item in pr_entries:
        ax.plot(
            item["recall"],
            item["precision"],
            color=item["color"],
            linewidth=2.5 if item["short"] == "baseline" else 2.1,
            label=f"{item['label']} (AP={item['ap']:.3f})",
        )
    ax.plot(
        [0, 1],
        [pr_entries[0]["pos_rate"]] * 2,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.1,
        alpha=0.45,
        label=f"Pos rate={pr_entries[0]['pos_rate']:.3f}",
    )
    style_axes(ax, "Recall", "Precision", "Test PR curves by attention heads")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "(c)", panel_label_size)

    ax = axes[1, 1]
    for item in roc_entries:
        ax.plot(
            item["fpr"],
            item["tpr"],
            color=item["color"],
            linewidth=2.5 if item["short"] == "baseline" else 2.1,
            label=f"{item['label']} (AUC={item['auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.1, label="Random chance")
    style_axes(ax, "False Positive Rate", "True Positive Rate", "Test ROC curves by attention heads")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "(d)", panel_label_size)

    plt.close(fig)

    panel_outputs = [
        ("fig_11a_attention_heads_val_auprc.png", "auprc"),
        ("fig_11b_attention_heads_val_auroc.png", "auroc"),
        ("fig_11c_attention_heads_test_pr.png", "pr"),
        ("fig_11d_attention_heads_test_roc.png", "roc"),
    ]
    for filename, panel in panel_outputs:
        single_fig, ax = make_single_panel()
        if panel == "auprc":
            for item in auprc_series:
                is_baseline = item["short"] == "baseline"
                ax.plot(
                    item["epochs"],
                    item["values"],
                    color=item["color"],
                    linewidth=3.4 if is_baseline else 2.9,
                    linestyle="-" if is_baseline else "--",
                    marker="o",
                    markersize=4.4,
                    label=item["label"],
                )
            style_axes(ax, "Epoch", "Validation AUPRC", "Validation AUPRC across epochs")
            ax.set_title("")
            top_vals = np.concatenate([item["values"] for item in auprc_series])
            pad = max((top_vals.max() - top_vals.min()) * 0.12, 0.01)
            ax.set_ylim(top_vals.min() - pad, top_vals.max() + pad * 0.35)
            ax.legend(frameon=False, loc="lower right")
        elif panel == "auroc":
            for item in auroc_series:
                is_baseline = item["short"] == "baseline"
                ax.plot(
                    item["epochs"],
                    item["values"],
                    color=item["color"],
                    linewidth=3.4 if is_baseline else 2.9,
                    linestyle="-" if is_baseline else "--",
                    marker="o",
                    markersize=4.4,
                    label=item["label"],
                )
            style_axes(ax, "Epoch", "Validation AUROC", "Validation AUROC across epochs")
            ax.set_title("")
            top_vals = np.concatenate([item["values"] for item in auroc_series])
            pad = max((top_vals.max() - top_vals.min()) * 0.12, 0.01)
            ax.set_ylim(top_vals.min() - pad, top_vals.max() + pad * 0.35)
            ax.legend(frameon=False, loc="lower right")
        elif panel == "pr":
            for item in pr_entries:
                ax.plot(
                    item["recall"],
                    item["precision"],
                    color=item["color"],
                    linewidth=3.5 if item["short"] == "baseline" else 3.0,
                    label=f"{item['label']} (AP={item['ap']:.3f})",
                )
            ax.plot(
                [0, 1],
                [pr_entries[0]["pos_rate"]] * 2,
                color="#9CA3AF",
                linestyle="--",
                linewidth=1.8,
                alpha=0.45,
                label=f"Pos rate={pr_entries[0]['pos_rate']:.3f}",
            )
            style_axes(ax, "Recall", "Precision", "Test PR curves by attention heads")
            ax.set_title("")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, loc="lower left")
        else:
            for item in roc_entries:
                ax.plot(
                    item["fpr"],
                    item["tpr"],
                    color=item["color"],
                    linewidth=3.5 if item["short"] == "baseline" else 3.0,
                    label=f"{item['label']} (AUC={item['auc']:.3f})",
            )
            ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.8, label="Random chance")
            style_axes(ax, "False Positive Rate", "True Positive Rate", "Test ROC curves by attention heads")
            ax.set_title("")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, loc="lower right")
        apply_perf_single_panel_fonts_11(ax)
        single_fig.tight_layout()
        single_fig.savefig(ATTENTION_OUTPUT_DIR / filename, dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
        plt.close(single_fig)


def main() -> None:
    args = parse_args()
    configure_style()

    if args.figure in {"all", "pic_human"}:
        plot_pic_human_summary(args.panel_label_size)
    if args.figure in {"all", "attention_heads"}:
        plot_attention_heads_summary(args.panel_label_size)


if __name__ == "__main__":
    main()
