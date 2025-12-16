#!/usr/bin/env python3
"""
Plot training/validation curves from PIC human CSV results.

Usage:
  python scripts/plot_pic_human_results.py \
    --csv result/model_train_results/PIC_human/PIC_human_val_result.csv \
    --outdir result/plots/PIC_human

Outputs:
  - loss_acc.png: Train/val loss and accuracy over epochs
  - val_metrics.png: Validation precision/recall/F1/AUC/PR-AUC over epochs
  - metrics_overview.png: 2x3 panel overview of all metrics
"""

import argparse
import os
from pathlib import Path

import matplotlib

# Use non-interactive backend to save files in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TOP_PAL = {
    "train": "#4D4D4D",  # dark grey
    "val": "#254E7B",  # primary blue
    "precision": "#254E7B",
    "recall": "#F08C2E",
    "f1": "#6AC1B8",
    "roc_auc": "#254E7B",
    "pr_auc": "#F08C2E",
}


def apply_pub_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "axes.linewidth": 1.8,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "grid.alpha": 0.25,
        }
    )


def style_axes(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(1.8)
    ax.tick_params(axis="both", which="both", width=1.6, length=5)


def _best_index(series: pd.Series, mode: str = "max"):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().all():
        return None
    if mode == "min":
        best_val = vals.min()
    else:
        best_val = vals.max()
    best_idx = int(vals[vals == best_val].index[0])
    return best_idx, float(best_val)


def annotate_best(
    ax,
    epochs,
    series: pd.Series,
    *,
    mode: str,
    color: str,
    fmt: str = ".3f",
    label_prefix: str = "Best",
    xytext: tuple[int, int] = (8, 10),
):
    res = _best_index(series, mode)
    if res is None:
        return
    idx, best_val = res
    x = epochs[idx]
    y = best_val
    ax.scatter([x], [y], s=55, color=color, edgecolors="white", linewidth=1.0, zorder=5)
    txt = f"{label_prefix} {best_val:{fmt}}"
    ax.annotate(
        txt,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        color=color,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0, alpha=0.85),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
    )


def _plot_best_trend(ax, epochs, values, color, label_suffix="Best so far"):
    values = pd.Series(values)
    if values.isna().all():
        return
    best = values.cummax()
    ax.plot(
        epochs,
        best,
        linestyle="--",
        linewidth=1.5,
        color=color,
        alpha=0.55,
        label=label_suffix,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Plot PIC human training results from CSV")
    p.add_argument("--csv", required=True, help="Path to PIC_human CSV file")
    p.add_argument(
        "--outdir",
        default="result/plots/PIC_human",
        help="Directory to save plots",
    )
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    p.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Image format",
    )
    p.add_argument(
        "--prefix",
        default="human",
        help="Column prefix used in the CSV (e.g. 'human', 'KMS-11').",
    )
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")


def plot_loss_acc(df: pd.DataFrame, prefix: str, outpath: Path, dpi: int, fmt: str):
    col = lambda name: f"{prefix}_{name}"
    epochs = df[col("epoch")].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Losses
    axes[0].plot(
        epochs,
        df[col("train_loss")],
        label="Train Loss",
        marker="o",
        linewidth=2.2,
        color=TOP_PAL["train"],
    )
    axes[0].plot(
        epochs,
        df[col("val_loss")],
        label="Val Loss",
        marker="o",
        linewidth=2.2,
        color=TOP_PAL["val"],
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    style_axes(axes[0])
    # annotate best (min) Val Loss
    if col("val_loss") in df.columns:
        annotate_best(
            axes[0],
            epochs,
            df[col("val_loss")],
            mode="min",
            color=TOP_PAL["val"],
            label_prefix="Min",
        )

    # Accuracy
    axes[1].plot(
        epochs,
        df[col("train_acc")],
        label="Train Acc",
        marker="o",
        linewidth=2.2,
        color=TOP_PAL["train"],
    )
    axes[1].plot(
        epochs,
        df[col("val_acc")],
        label="Val Acc",
        marker="o",
        linewidth=2.2,
        color=TOP_PAL["val"],
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    style_axes(axes[1])
    # annotate best (max) Val Acc
    if col("val_acc") in df.columns:
        annotate_best(axes[1], epochs, df[col("val_acc")], mode="max", color=TOP_PAL["val"])

    outfile = outpath / f"loss_acc.{fmt}"
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_val_metrics(df: pd.DataFrame, prefix: str, outpath: Path, dpi: int, fmt: str):
    col = lambda name: f"{prefix}_{name}"
    epochs = df[col("epoch")].values
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    lines = []
    metrics = [
        (col("val_prec"), "Precision"),
        (col("val_recall"), "Recall"),
        (col("val_f1"), "F1"),
        (col("val_auc"), "ROC AUC"),
        (col("val_pr_auc"), "PR AUC"),
    ]
    for col_name, label in metrics:
        if col_name in df.columns:
            color = {
                "Precision": TOP_PAL["precision"],
                "Recall": TOP_PAL["recall"],
                "F1": TOP_PAL["f1"],
                "ROC AUC": TOP_PAL["roc_auc"],
                "PR AUC": TOP_PAL["pr_auc"],
            }[label]
            values = df[col_name]
            (ln,) = ax.plot(epochs, values, marker="o", label=label, linewidth=2.2, color=color)
            lines.append(ln)
            if label in {"ROC AUC", "PR AUC"}:
                _plot_best_trend(ax, epochs, values, color)
            annotate_best(ax, epochs, values, mode="max", color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics over Epochs")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    style_axes(ax)

    outfile = outpath / f"val_metrics.{fmt}"
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def _calc_ylim(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        pad = 0.02
        return max(0.0, vmin - pad), min(1.0, vmax + pad)
    # add a 5% padding around the range, clipped to [0,1]
    pad = (vmax - vmin) * 0.05
    ymin = max(0.0, vmin - pad)
    ymax = min(1.0, vmax + pad)
    # Ensure we do not start exactly at 0 unless necessary
    if ymin == 0.0 and vmin > 0.02:
        ymin = max(0.0, vmin - max(pad, 0.02))
    return ymin, ymax


def plot_each_val_metric(df: pd.DataFrame, prefix: str, outpath: Path, dpi: int, fmt: str):
    col = lambda name: f"{prefix}_{name}"
    epochs = df[col("epoch")].values
    metric_map = {
        col("val_prec"): "val_precision",
        col("val_recall"): "val_recall",
        col("val_f1"): "val_f1",
        col("val_auc"): "val_auc",
        col("val_pr_auc"): "val_pr_auc",
    }

    for col_name, name in metric_map.items():
        if col_name not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        title = {
            "val_precision": "Validation Precision",
            "val_recall": "Validation Recall",
            "val_f1": "Validation F1",
            "val_auc": "Validation ROC AUC",
            "val_pr_auc": "Validation PR AUC",
        }.get(name, name)
        ax.set_title(title)
        color = {
            "val_precision": TOP_PAL["precision"],
            "val_recall": TOP_PAL["recall"],
            "val_f1": TOP_PAL["f1"],
            "val_auc": TOP_PAL["roc_auc"],
            "val_pr_auc": TOP_PAL["pr_auc"],
        }[name]
        values = df[col_name]
        ax.plot(epochs, values, marker="o", linewidth=2.2, color=color)
        if name in {"val_auc", "val_pr_auc"}:
            _plot_best_trend(ax, epochs, values, color)
        ylim = _calc_ylim(values)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        style_axes(ax)
        annotate_best(ax, epochs, values, mode="max", color=color)
        outfile = outpath / f"{name}.{fmt}"
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)


def plot_overview(df: pd.DataFrame, prefix: str, outpath: Path, dpi: int, fmt: str):
    col = lambda name: f"{prefix}_{name}"
    epochs = df[col("epoch")].values

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    # Losses
    axes[0].plot(epochs, df[col("train_loss")], label="Train", marker="o")
    axes[0].plot(epochs, df[col("val_loss")], label="Val", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, df[col("train_acc")], label="Train", marker="o")
    axes[1].plot(epochs, df[col("val_acc")], label="Val", marker="o")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Acc")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Precision
    axes[2].plot(
        epochs,
        df.get(col("val_prec"), pd.Series(index=df.index)),
        marker="o",
        linewidth=2.0,
        color=TOP_PAL["precision"],
    )
    axes[2].set_title("Val Precision")
    axes[2].set_xlabel("Epoch")
    ylim = _calc_ylim(df.get(col("val_prec"), pd.Series(index=df.index)))
    if ylim is not None:
        axes[2].set_ylim(*ylim)
    axes[2].grid(True, alpha=0.3)
    style_axes(axes[2])
    if col("val_prec") in df.columns:
        annotate_best(axes[2], epochs, df[col("val_prec")], mode="max", color=TOP_PAL["precision"])

    # Recall
    axes[3].plot(
        epochs,
        df.get(col("val_recall"), pd.Series(index=df.index)),
        marker="o",
        linewidth=2.0,
        color=TOP_PAL["recall"],
    )
    axes[3].set_title("Val Recall")
    axes[3].set_xlabel("Epoch")
    ylim = _calc_ylim(df.get(col("val_recall"), pd.Series(index=df.index)))
    if ylim is not None:
        axes[3].set_ylim(*ylim)
    axes[3].grid(True, alpha=0.3)
    style_axes(axes[3])
    if col("val_recall") in df.columns:
        annotate_best(axes[3], epochs, df[col("val_recall")], mode="max", color=TOP_PAL["recall"])

    # F1
    axes[4].plot(
        epochs,
        df.get(col("val_f1"), pd.Series(index=df.index)),
        marker="o",
        linewidth=2.0,
        color=TOP_PAL["f1"],
    )
    axes[4].set_title("Val F1")
    axes[4].set_xlabel("Epoch")
    ylim = _calc_ylim(df.get(col("val_f1"), pd.Series(index=df.index)))
    if ylim is not None:
        axes[4].set_ylim(*ylim)
    axes[4].grid(True, alpha=0.3)
    style_axes(axes[4])
    if col("val_f1") in df.columns:
        annotate_best(axes[4], epochs, df[col("val_f1")], mode="max", color=TOP_PAL["f1"])

    # PR AUC or ROC AUC
    if col("val_pr_auc") in df.columns:
        values = df[col("val_pr_auc")]
        axes[5].plot(
            epochs,
            values,
            marker="o",
            linewidth=2.0,
            color=TOP_PAL["pr_auc"],
        )
        _plot_best_trend(axes[5], epochs, values, TOP_PAL["pr_auc"])
        axes[5].set_title("Val PR AUC")
    elif col("val_auc") in df.columns:
        values = df[col("val_auc")]
        axes[5].plot(
            epochs,
            values,
            marker="o",
            linewidth=2.0,
            color=TOP_PAL["roc_auc"],
        )
        _plot_best_trend(axes[5], epochs, values, TOP_PAL["roc_auc"])
        axes[5].set_title("Val ROC AUC")
    else:
        axes[5].set_title("Val AUC")
    axes[5].set_xlabel("Epoch")
    # choose the present series for ylim
    if col("val_pr_auc") in df.columns:
        ylim = _calc_ylim(df[col("val_pr_auc")])
    elif col("val_auc") in df.columns:
        ylim = _calc_ylim(df[col("val_auc")])
    else:
        ylim = None
    if ylim is not None:
        axes[5].set_ylim(*ylim)
    axes[5].grid(True, alpha=0.3)
    style_axes(axes[5])
    if col("val_pr_auc") in df.columns:
        annotate_best(
            axes[5],
            epochs,
            df[col("val_pr_auc")],
            mode="max",
            color=TOP_PAL["pr_auc"],
            xytext=(8, -16),
        )
    elif col("val_auc") in df.columns:
        annotate_best(
            axes[5],
            epochs,
            df[col("val_auc")],
            mode="max",
            color=TOP_PAL["roc_auc"],
            xytext=(8, -16),
        )

    outfile = outpath / f"metrics_overview.{fmt}"
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    prefix = args.prefix.strip()
    col = lambda name: f"{prefix}_{name}"

    # Ensure required columns exist
    ensure_columns(
        df,
        [
            col("epoch"),
            col("train_loss"),
            col("train_acc"),
            col("val_loss"),
            col("val_acc"),
        ],
    )

    # Sort by epoch in case CSV is unsorted
    df = df.sort_values(col("epoch")).reset_index(drop=True)

    apply_pub_style()
    plot_loss_acc(df, prefix, outdir, dpi=args.dpi, fmt=args.format)
    plot_val_metrics(df, prefix, outdir, dpi=args.dpi, fmt=args.format)
    plot_each_val_metric(df, prefix, outdir, dpi=args.dpi, fmt=args.format)
    plot_overview(df, prefix, outdir, dpi=args.dpi, fmt=args.format)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
