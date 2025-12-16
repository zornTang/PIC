#!/usr/bin/env python3
"""
Export per-sample predictions (y_true, y_score) for PIC model on val/test sets.

This enables downstream plotting of PR/ROC, calibration, and confusion matrix.

Example:
  python scripts/export_pic_predictions.py \
    --model result/model_train_results/PIC_human/PIC_human_model.pth \
    --label_name human \
    --data_path data/human_data.pkl \
    --feature_dir result/seq_embedding \
    --outdir result/predictions \
    --device cpu
"""

import argparse
from pathlib import Path
import os
import json
import sys
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd

# Ensure local `code/` package takes precedence over stdlib `code` module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from module.load_dataset import get_index, PIC_Dataset
from module.PIC import PIC


def load_model_config(model_path: str) -> dict:
    config_path = Path(model_path).parent / "model_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Warning: {config_path} not found. Falling back to CLI hyperparameters.")
    return {}


def build_model_from_config(config: dict, args, device: torch.device) -> PIC:
    return PIC(
        input_shape=config.get("input_size", args.input_size),
        output_shape=config.get("output_size", args.output_size),
        hidden_units=config.get("hidden_size", args.hidden_size),
        attn_drop=config.get("attn_drop", args.attn_drop),
        linear_drop=config.get("linear_drop", args.linear_drop),
        device=device,
        model_variant=config.get("model_variant", "attention"),
        num_heads=config.get("num_heads", 1),
        cnn_channels=config.get("cnn_channels", 256),
        cnn_kernel_size=config.get("cnn_kernel_size", 5),
        cnn_layers=config.get("cnn_layers", 2),
        cnn_drop=config.get("cnn_drop", 0.1),
    )


def remap_legacy_attention_keys(state_dict):
    """Map old attention keys (without backbone prefix) to new names."""
    needs_remap = any(
        key.startswith("multihead_attention.") or key.startswith("layerNorm.")
        for key in state_dict.keys()
    )
    if not needs_remap:
        return state_dict
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("multihead_attention."):
            new_key = "backbone." + key
        elif key.startswith("layerNorm."):
            suffix = key.split(".", 1)[1]
            new_key = f"backbone.layer_norm.{suffix}"
        remapped[new_key] = value
    return remapped


def parse_args():
    p = argparse.ArgumentParser(description="Export PIC predictions for val/test sets")
    p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    p.add_argument("--label_name", required=True, help="Label name (e.g., human)")
    p.add_argument("--data_path", required=True, help="Pickle dataset path (e.g., data/human_data.pkl)")
    p.add_argument("--feature_dir", required=True, help="ESM embedding dir (e.g., result/seq_embedding)")
    p.add_argument("--outdir", default="result/predictions", help="Output directory for CSVs")
    p.add_argument("--device", default="cpu", help="cpu or cuda:N")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--max_length", type=int, default=1000)
    p.add_argument("--feature_length", type=int, default=1280)
    p.add_argument("--input_size", type=int, default=1280)
    p.add_argument("--hidden_size", type=int, default=320)
    p.add_argument("--output_size", type=int, default=1)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--split", choices=["val", "test", "both"], default="val")
    p.add_argument("--linear_drop", type=float, default=0.1)
    p.add_argument("--attn_drop", type=float, default=0.3)
    return p.parse_args()


def run_export(args):
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_model_config(args.model)

    max_length = config.get("max_length", args.max_length)
    feature_length = config.get("feature_length", args.feature_length)
    input_size = config.get("input_size", args.input_size)
    hidden_size = config.get("hidden_size", args.hidden_size)
    output_size = config.get("output_size", args.output_size)
    batch_size = config.get("batch_size", args.batch_size)
    test_ratio = config.get("test_ratio", args.test_ratio)
    val_ratio = config.get("val_ratio", args.val_ratio)
    random_seed = config.get("random_seed", args.random_seed)

    # Reconstruct splits (must match training args)
    _, val_dict, test_indexes = get_index(
        data_path=args.data_path,
        label_name=args.label_name,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    val_indexes = [list(d.keys())[0] for d in val_dict]

    # Build datasets
    def make_dataset(index_list):
        return PIC_Dataset(
            indexes=index_list,
            feature_dir=args.feature_dir,
            label_name=args.label_name,
            max_length=max_length,
            feature_length=feature_length,
            device=device,
        )

    to_run = []
    if args.split in ("val", "both"):
        to_run.append(("val", make_dataset(val_indexes)))
    if args.split in ("test", "both"):
        to_run.append(("test", make_dataset(test_indexes)))

    # Model
    model = build_model_from_config(config, args, device).to(device)
    state = torch.load(args.model, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        # Handle legacy checkpoints without backbone prefix
        remapped_state = remap_legacy_attention_keys(state)
        if remapped_state is state:
            raise err
        model.load_state_dict(remapped_state)
    model.eval()

    for split_name, dataset in to_run:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ys_true = []
        ys_score = []
        with torch.inference_mode():
            for X, y, pad_idx in loader:
                logits = model(X.to(device), pad_idx.to(device))
                score = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
                ys_score.append(score)
                ys_true.append(y.detach().cpu().numpy())
        y_true = np.concatenate(ys_true).astype(int).ravel()
        y_score = np.concatenate(ys_score).astype(float).ravel()
        df_out = pd.DataFrame({"y_true": y_true, "y_score": y_score})
        out_path = outdir / f"PIC_{args.label_name}_{split_name}_predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved {split_name} predictions: {out_path} (N={len(df_out)})")


if __name__ == "__main__":
    args = parse_args()
    run_export(args)
