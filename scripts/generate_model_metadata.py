#!/usr/bin/env python3
"""
Generate model_config.json and cached prediction arrays (val/test) for
an already-trained PIC model. This is useful for older checkpoints
that were produced before the new metadata/ablation workflow.

Example:
  python scripts/generate_model_metadata.py \
    --model_dir result/model_train_results/PIC_human \
    --model_name PIC_human_model.pth \
    --data_path data/human_data.pkl \
    --label_name human \
    --feature_dir result/seq_embedding
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure local code/ package has priority over stdlib `code`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from module.PIC import PIC
from module.load_dataset import get_index, PIC_Dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate model_config.json and prediction caches for a PIC checkpoint."
    )
    parser.add_argument("--model_dir", required=True, help="Directory containing the trained model.")
    parser.add_argument("--model_name", default="PIC_human_model.pth", help="Checkpoint filename.")
    parser.add_argument("--data_path", required=True, help="Path to the pickled training dataset.")
    parser.add_argument("--label_name", required=True, help="Label column (e.g., human).")
    parser.add_argument("--feature_dir", required=True, help="Directory holding ESM embeddings.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda:N).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=1000, help="Sequence truncation length.")
    parser.add_argument("--feature_length", type=int, default=1280, help="Embedding dimension.")
    parser.add_argument("--input_size", type=int, default=1280, help="Model input size.")
    parser.add_argument("--hidden_size", type=int, default=320, help="Hidden units for MLP.")
    parser.add_argument("--output_size", type=int, default=1, help="Model output size.")
    parser.add_argument("--linear_drop", type=float, default=0.1, help="Dropout in linear layers.")
    parser.add_argument("--attn_drop", type=float, default=0.3, help="Dropout in attention layer.")
    parser.add_argument("--model_variant", choices=["attention", "cnn", "avgpool"],
                        default="attention", help="Backbone variant.")
    parser.add_argument("--num_heads", type=int, default=1, help="Attention heads (if applicable).")
    parser.add_argument("--cnn_channels", type=int, default=256, help="CNN channels (cnn variant).")
    parser.add_argument("--cnn_kernel_size", type=int, default=5, help="CNN kernel size.")
    parser.add_argument("--cnn_layers", type=int, default=2, help="Number of CNN layers.")
    parser.add_argument("--cnn_drop", type=float, default=0.1, help="CNN dropout.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splits.")
    return parser.parse_args()


def build_model(cfg, device):
    model = PIC(
        input_shape=cfg["input_size"],
        output_shape=cfg["output_size"],
        hidden_units=cfg["hidden_size"],
        attn_drop=cfg["attn_drop"],
        linear_drop=cfg["linear_drop"],
        device=device,
        model_variant=cfg["model_variant"],
        num_heads=cfg["num_heads"],
        cnn_channels=cfg["cnn_channels"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        cnn_layers=cfg["cnn_layers"],
        cnn_drop=cfg["cnn_drop"],
    )
    return model


def collect_predictions(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, targets = [], []
    model.eval()
    with torch.inference_mode():
        for X, y, pad_idx in loader:
            logits = model(X.to(device), pad_idx.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.append(y.cpu().numpy().ravel())
    return np.concatenate(probs), np.concatenate(targets)


def remap_legacy_attention_keys(state_dict):
    """Map old (pre-backbone) attention keys to the new module names."""
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


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device(args.device)
    cfg = {
        "data_path": args.data_path,
        "label_name": args.label_name,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "feature_dir": args.feature_dir,
        "batch_size": args.batch_size,
        "linear_drop": args.linear_drop,
        "attn_drop": args.attn_drop,
        "max_length": args.max_length,
        "feature_length": args.feature_length,
        "learning_rate": None,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "output_size": args.output_size,
        "device": args.device,
        "num_epochs": None,
        "random_seed": args.random_seed,
        "model_variant": args.model_variant,
        "num_heads": args.num_heads,
        "cnn_channels": args.cnn_channels,
        "cnn_kernel_size": args.cnn_kernel_size,
        "cnn_layers": args.cnn_layers,
        "cnn_drop": args.cnn_drop,
        "param_count": None,
    }

    model = build_model(cfg, device).to(device)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        # Attempt remap for legacy attention checkpoints
        remapped_state = remap_legacy_attention_keys(state)
        if remapped_state is state:
            raise err
        model.load_state_dict(remapped_state)
    cfg["param_count"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    _, val_dict, test_indexes = get_index(
        data_path=cfg["data_path"],
        label_name=cfg["label_name"],
        test_ratio=cfg["test_ratio"],
        val_ratio=cfg["val_ratio"],
        random_seed=cfg["random_seed"],
    )
    val_indexes = [list(d.keys())[0] for d in val_dict]

    val_dataset = PIC_Dataset(
        indexes=val_indexes,
        feature_dir=cfg["feature_dir"],
        label_name=cfg["label_name"],
        max_length=cfg["max_length"],
        feature_length=cfg["feature_length"],
        device=device,
    )
    test_dataset = PIC_Dataset(
        indexes=test_indexes,
        feature_dir=cfg["feature_dir"],
        label_name=cfg["label_name"],
        max_length=cfg["max_length"],
        feature_length=cfg["feature_length"],
        device=device,
    )

    val_scores, val_targets = collect_predictions(model, val_dataset, cfg["batch_size"], device)
    test_scores, test_targets = collect_predictions(model, test_dataset, cfg["batch_size"], device)

    np.save(model_dir / "val_pred_scores.npy", val_scores)
    np.save(model_dir / "val_targets.npy", val_targets)
    np.save(model_dir / "test_pred_scores.npy", test_scores)
    np.save(model_dir / "test_targets.npy", test_targets)

    config_path = model_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Metadata and prediction caches written to {model_dir}")


if __name__ == "__main__":
    main()
