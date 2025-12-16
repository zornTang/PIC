#!/usr/bin/env python3
"""
Aggregate AUROC/AUPRC metrics for multiple ablation runs and
estimate whether each variant significantly differs from the baseline.

Each experiment directory must contain:
  - model_config.json
  - val_pred_scores.npy / val_targets.npy
  - test_pred_scores.npy / test_targets.npy

Usage example:
  python analysis/scripts/ablation_evaluation.py \
    --experiments baseline=result/model_train_results/PIC_human \
                 no_attention=result/ablations/no_attention/PIC_human \
                 cnn=result/ablations/cnn/PIC_human \
    --baseline baseline \
    --splits val test \
    --output_dir analysis/results/ablations \
    --bootstrap_rounds 2000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compare AUROC/AUPRC across ablation experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="List of NAME=PATH entries. PATH should point to a PIC_xxx run directory.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Name of the experiment in --experiments to be used as baseline.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Evaluation splits to include.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results/ablations",
        help="Directory to store summary tables.",
    )
    parser.add_argument(
        "--bootstrap_rounds",
        type=int,
        default=1000,
        help="Number of bootstrap samples for significance testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling.",
    )
    return parser.parse_args()


def load_split_arrays(run_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    score_path = run_dir / f"{split}_pred_scores.npy"
    target_path = run_dir / f"{split}_targets.npy"
    if not score_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing prediction files for split '{split}' in {run_dir}")
    scores = np.load(score_path)
    targets = np.load(target_path)
    return scores.reshape(-1), targets.reshape(-1)


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"{config_path} not found. Cannot determine model metadata.")


def safe_metric(metric_fn, y_true, y_score):
    try:
        return metric_fn(y_true, y_score)
    except ValueError:
        return np.nan


def bootstrap_pvalue(y_true, scores_a, scores_b, metric_fn, rounds, rng):
    observed = safe_metric(metric_fn, y_true, scores_a) - safe_metric(metric_fn, y_true, scores_b)
    if np.isnan(observed):
        return np.nan, np.nan
    diffs = []
    n = len(y_true)
    for _ in range(rounds):
        sample_idx = rng.integers(0, n, size=n)
        sample_true = y_true[sample_idx]
        sample_a = scores_a[sample_idx]
        sample_b = scores_b[sample_idx]
        diff = safe_metric(metric_fn, sample_true, sample_a) - safe_metric(metric_fn, sample_true, sample_b)
        if not np.isnan(diff):
            diffs.append(diff)
    if not diffs:
        return observed, np.nan
    diffs = np.array(diffs)
    p_value = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (len(diffs) + 1)
    return observed, p_value


def main():
    args = parse_args()
    experiments = {}
    for exp_entry in args.experiments:
        if "=" not in exp_entry:
            raise ValueError(f"Experiment entry '{exp_entry}' must be in NAME=PATH format.")
        name, path = exp_entry.split("=", 1)
        experiments[name] = Path(path).expanduser().resolve()

    if args.baseline not in experiments:
        raise ValueError(f"Baseline '{args.baseline}' not found in provided experiments.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_data: Dict[Tuple[str, str], Dict] = {}
    records = []

    for name, run_dir in experiments.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Experiment path {run_dir} does not exist.")
        config = load_config(run_dir)
        param_count = config.get("param_count", np.nan)
        model_variant = config.get("model_variant", "attention")

        for split in args.splits:
            scores, targets = load_split_arrays(run_dir, split)
            auroc = safe_metric(roc_auc_score, targets, scores)
            auprc = safe_metric(average_precision_score, targets, scores)
            record = {
                "experiment": name,
                "run_dir": str(run_dir),
                "split": split,
                "num_samples": len(targets),
                "positive_rate": float(np.mean(targets)) if len(targets) > 0 else np.nan,
                "auroc": auroc,
                "auprc": auprc,
                "model_variant": model_variant,
                "param_count": param_count,
            }
            records.append(record)
            exp_data[(name, split)] = {
                "scores": scores,
                "targets": targets,
                "auroc": auroc,
                "auprc": auprc,
            }

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise RuntimeError("No evaluation data collected. Check experiment paths and splits.")

    results_df["delta_auroc_vs_baseline"] = np.nan
    results_df["delta_auprc_vs_baseline"] = np.nan
    results_df["pvalue_auroc"] = np.nan
    results_df["pvalue_auprc"] = np.nan

    rng = np.random.default_rng(args.seed)
    baseline_name = args.baseline

    for name in experiments:
        if name == baseline_name:
            continue
        for split in args.splits:
            key = (name, split)
            baseline_key = (baseline_name, split)
            if key not in exp_data or baseline_key not in exp_data:
                continue
            comp = exp_data[key]
            base = exp_data[baseline_key]
            if len(comp["targets"]) != len(base["targets"]) or not np.array_equal(comp["targets"], base["targets"]):
                raise ValueError(
                    f"Targets mismatch between baseline '{baseline_name}' and experiment '{name}' for split '{split}'."
                )
            auroc_delta, auroc_p = bootstrap_pvalue(
                y_true=base["targets"],
                scores_a=comp["scores"],
                scores_b=base["scores"],
                metric_fn=roc_auc_score,
                rounds=args.bootstrap_rounds,
                rng=rng,
            )
            auprc_delta, auprc_p = bootstrap_pvalue(
                y_true=base["targets"],
                scores_a=comp["scores"],
                scores_b=base["scores"],
                metric_fn=average_precision_score,
                rounds=args.bootstrap_rounds,
                rng=rng,
            )
            mask = (results_df["experiment"] == name) & (results_df["split"] == split)
            results_df.loc[mask, "delta_auroc_vs_baseline"] = auroc_delta
            results_df.loc[mask, "delta_auprc_vs_baseline"] = auprc_delta
            results_df.loc[mask, "pvalue_auroc"] = auroc_p
            results_df.loc[mask, "pvalue_auprc"] = auprc_p

    csv_path = output_dir / "ablation_metrics.csv"
    json_path = output_dir / "ablation_metrics.json"
    results_df.sort_values(["split", "experiment"]).to_csv(csv_path, index=False)
    results_df.to_json(json_path, orient="records", indent=2)
    print(f"Saved ablation summary to {csv_path}")


if __name__ == "__main__":
    main()
