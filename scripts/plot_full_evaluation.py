#!/usr/bin/env python3
"""
Full evaluation plotting from per-sample predictions.

Inputs: a CSV with columns at minimum
  - y_true: {0,1}
  - y_score: model probability/score in [0,1]
Optional:
  - id: sample identifier

Outputs to --outdir:
  - panel.png/svg/pdf: Composite figure (PR main, ROC, Calibration, Confusion matrix)
  - pr_curve.png, roc_curve.png, calibration.png, confusion_matrix.png
  - summary.csv: N, prevalence, AUPRC [CI], AUROC [CI], ECE, Brier, and workpoint metrics w/ CI

Usage example:
  MPLCONFIGDIR=.matplotlib_cache \
  python scripts/plot_full_evaluation.py \
    --pred result/predictions/PIC_human_val_predictions.csv \
    --outdir result/plots/PIC_human/eval \
    --workpoint best_f1 \
    --boot 500
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# Publication-like style and palette
TOP_PAL = {
    "precision": "#1f77b4",  # blue
    "recall": "#ff7f0e",  # orange
    "f1": "#2ca02c",      # green
    "roc_auc": "#d62728",  # red
    "pr_auc": "#9467bd",   # purple
    "cm": "#2c7fb8",       # blue heatmap base
}


def apply_pub_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.linewidth": 1.8,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "grid.alpha": 0.25,
        }
    )
    sns.set_context("talk")


def style_axes(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(1.8)
    ax.tick_params(axis="both", which="both", width=1.6, length=5)


def bootstrap_metric_ci(y_true, y_score, fn, n_boot=500, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        try:
            stats.append(fn(yt, ys))
        except Exception:
            continue
    if not stats:
        return np.nan, (np.nan, np.nan)
    stats = np.array(stats)
    est = fn(y_true, y_score)
    low = np.percentile(stats, (100 - ci) / 2)
    high = np.percentile(stats, 100 - (100 - ci) / 2)
    return est, (low, high)


def compute_best_f1_threshold(y_true, y_score):
    p, r, thr = precision_recall_curve(y_true, y_score)
    # scikit returns thresholds for all but the last point
    f1 = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thr[i])


def compute_workpoint_metrics(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
    mcc = mcc_num / (mcc_den + 1e-12)
    balanced_acc = 0.5 * (recall + specificity)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "MCC": mcc,
        "BalancedAcc": balanced_acc,
    }


def bootstrap_workpoint_ci(y_true, y_score, thr, n_boot=500, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        stats.append(compute_workpoint_metrics(yt, ys, thr))
    # Aggregate CIs
    out = {}
    keys = [k for k in stats[0].keys() if k not in ("TP", "FP", "FN", "TN")]
    for k in keys:
        vals = np.array([d[k] for d in stats])
        low = np.percentile(vals, (100 - ci) / 2)
        high = np.percentile(vals, 100 - (100 - ci) / 2)
        out[k] = (low, high)
    return out


def calibration_bins(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_score, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)
        else:
            ys = y_score[mask]
            yt = y_true[mask]
            bin_acc.append(np.mean(yt))
            bin_conf.append(np.mean(ys))
            bin_count.append(np.sum(mask))
    return np.array(bin_acc), np.array(bin_conf), np.array(bin_count), bins


def ece_score(y_true, y_score, n_bins=10):
    acc, conf, cnt, _ = calibration_bins(y_true, y_score, n_bins)
    n = len(y_true)
    valid = ~np.isnan(acc)
    w = cnt[valid] / n
    return np.sum(w * np.abs(acc[valid] - conf[valid]))


def brier_score(y_true, y_score):
    y_true = y_true.astype(float)
    return float(np.mean((y_score - y_true) ** 2))


def plot_pr(ax, y_true, y_score, work_thr=None, title_suffix="", color=TOP_PAL["pr_auc" ]):
    p, r, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    ax.plot(r, p, color=color, linewidth=2.2, label=f"PR (AUPRC={auprc:.3f})")
    prevalence = y_true.mean()
    ax.hlines(prevalence, 0, 1, colors="#999999", linestyles="--", label=f"Baseline={prevalence:.3f}")
    if work_thr is not None:
        # find point at thr
        y_pred = (y_score >= work_thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        ax.scatter([recall], [precision], s=65, color="#333333", edgecolors="white", zorder=5, label="Workpoint")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Precision–Recall{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    style_axes(ax)


def plot_roc(ax, y_true, y_score, color=TOP_PAL["roc_auc"]):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"ROC (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)


def plot_calibration(ax, y_true, y_score, n_bins=10):
    acc, conf, cnt, bins = calibration_bins(y_true, y_score, n_bins)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.plot(conf, acc, marker="o", linewidth=2.0, color="#2ca02c")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration (Reliability)")
    ax.grid(True, alpha=0.3)
    style_axes(ax)


def plot_confusion(ax, y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cmn,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (normalized)")
    style_axes(ax)


@dataclass
class Args:
    pred: Path
    outdir: Path
    workpoint: str
    thr: Optional[float]
    boot: int
    seed: int
    n_bins: int
    format: str


def parse_args():
    p = argparse.ArgumentParser(description="Full evaluation plots from per-sample predictions")
    p.add_argument("--pred", required=True, help="CSV with y_true,y_score[,id]")
    p.add_argument("--outdir", default="result/plots/PIC_human/eval", help="Output directory")
    p.add_argument("--workpoint", default="best_f1", choices=["best_f1", "fixed", "youden"], help="Workpoint selection")
    p.add_argument("--thr", type=float, default=None, help="Fixed threshold when workpoint=fixed")
    p.add_argument("--boot", type=int, default=500, help="Bootstrap iterations for CI")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n_bins", type=int, default=10, help="Calibration bins")
    p.add_argument("--format", default="png", choices=["png", "svg", "pdf"], help="Figure format")
    a = p.parse_args()
    return Args(
        pred=Path(a.pred),
        outdir=Path(a.outdir),
        workpoint=a.workpoint,
        thr=a.thr,
        boot=a.boot,
        seed=a.seed,
        n_bins=a.n_bins,
        format=a.format,
    )


def main():
    args = parse_args()
    apply_pub_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pred)
    if not {"y_true", "y_score"}.issubset(df.columns):
        raise SystemExit("CSV must contain columns: y_true, y_score")
    y_true = df["y_true"].to_numpy().astype(int)
    y_score = pd.to_numeric(df["y_score"], errors="coerce").to_numpy()
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    N = len(y_true)
    prevalence = float(np.mean(y_true))

    # Workpoint
    if args.workpoint == "best_f1":
        thr = compute_best_f1_threshold(y_true, y_score)
    elif args.workpoint == "youden":
        fpr, tpr, thr_roc = roc_curve(y_true, y_score)
        j = tpr - fpr
        thr = float(thr_roc[int(np.argmax(j))])
    else:
        if args.thr is None:
            raise SystemExit("--thr is required when --workpoint=fixed")
        thr = float(args.thr)

    # Summary metrics + CIs
    auprc, auprc_ci = bootstrap_metric_ci(y_true, y_score, average_precision_score, n_boot=args.boot, seed=args.seed)
    auroc, auroc_ci = bootstrap_metric_ci(y_true, y_score, roc_auc_score, n_boot=args.boot, seed=args.seed)
    ece = ece_score(y_true, y_score, n_bins=args.n_bins)
    brier = brier_score(y_true, y_score)
    wp = compute_workpoint_metrics(y_true, y_score, thr)
    wp_ci = bootstrap_workpoint_ci(y_true, y_score, thr, n_boot=args.boot, seed=args.seed)

    # PR main (left big)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax_pr = fig.add_subplot(gs[:, :2])
    ax_roc = fig.add_subplot(gs[0, 2])
    ax_cal = fig.add_subplot(gs[1, 2])
    ax_cm = fig.add_subplot(gs[2, 2])

    plot_pr(ax_pr, y_true, y_score, work_thr=thr)
    plot_roc(ax_roc, y_true, y_score)
    plot_calibration(ax_cal, y_true, y_score, n_bins=args.n_bins)
    plot_confusion(ax_cm, y_true, y_score, thr)

    # Add concise summary text bar
    summary_text = (
        f"N={N} | Prev={prevalence:.3f} | "
        f"AUPRC={auprc:.3f} [{auprc_ci[0]:.3f},{auprc_ci[1]:.3f}] | "
        f"AUROC={auroc:.3f} [{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}] | "
        f"ECE={ece:.3f} | Brier={brier:.3f} | "
        f"WP thr={thr:.3f} | P={wp['Precision']:.3f} R={wp['Recall']:.3f} F1={wp['F1']:.3f}"
    )
    fig.suptitle("Model Evaluation Summary", y=0.98)
    fig.text(0.01, 0.01, summary_text, fontsize=11)

    panel_path = args.outdir / f"panel.{args.format}"
    fig.savefig(panel_path)
    plt.close(fig)

    # Save individual plots
    for name, plotter in [
        ("pr_curve", lambda ax: plot_pr(ax, y_true, y_score, work_thr=thr)),
        ("roc_curve", lambda ax: plot_roc(ax, y_true, y_score)),
        ("calibration", lambda ax: plot_calibration(ax, y_true, y_score, n_bins=args.n_bins)),
    ]:
        fig_i, ax_i = plt.subplots(figsize=(6, 4), constrained_layout=True)
        plotter(ax_i)
        fig_i.savefig(args.outdir / f"{name}.{args.format}")
        plt.close(fig_i)

    # Confusion matrix alone
    fig_cm, ax_cm2 = plt.subplots(figsize=(5, 4), constrained_layout=True)
    plot_confusion(ax_cm2, y_true, y_score, thr)
    fig_cm.savefig(args.outdir / f"confusion_matrix.{args.format}")
    plt.close(fig_cm)

    # Save summary CSV
    flat = {
        "N": N,
        "Prevalence": prevalence,
        "AUPRC": auprc,
        "AUPRC_CI_low": auprc_ci[0],
        "AUPRC_CI_high": auprc_ci[1],
        "AUROC": auroc,
        "AUROC_CI_low": auroc_ci[0],
        "AUROC_CI_high": auroc_ci[1],
        "ECE": ece,
        "Brier": brier,
        "WP_threshold": thr,
        **{f"WP_{k}": v for k, v in wp.items() if k in ("Precision", "Recall", "F1", "Specificity", "MCC", "BalancedAcc")},
    }
    # append CI columns for workpoint
    for k, (lo, hi) in wp_ci.items():
        flat[f"WP_{k}_CI_low"] = lo
        flat[f"WP_{k}_CI_high"] = hi
    pd.DataFrame([flat]).to_csv(args.outdir / "summary.csv", index=False)

    print(f"Saved evaluation panel and metrics to: {args.outdir}")


if __name__ == "__main__":
    main()

