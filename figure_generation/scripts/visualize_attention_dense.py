#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _fg_paths import FIGURES_DIR, PIC2_INTEGRATED_JSON, PIC2_MODEL_DIR, PIC2_SEQ_EMBED_DIR
from module.PIC import PIC
from module.load_dataset import PIC_Dataset
from style_config import CMAPS, FIGSIZE, PALETTE, apply_style

apply_style()

NATURE_COLORS = {
    "primary_red": PALETTE["sig"],
    "primary_blue": PALETTE["human"],
    "primary_green": PALETTE["common"],
    "primary_orange": PALETTE["immune"],
    "primary_purple": "#8491B4",
    "dark_blue": PALETTE["human"],
    "dark_red": PALETTE["sig"],
}

HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

PROTEINS = [
    {"gene": "ATP6V1A", "index": 4914, "domains": [(45, 603, "V-ATPase A consensus")]},
    {"gene": "ATP6V1B2", "index": 4920, "domains": [(76, 499, "V-ATPase B consensus")]},
    {"gene": "H2BC11", "index": 23413, "domains": [(31, 125, "Histone H2B")]},
    {"gene": "PLBD1", "index": 99999, "domains": [(124, 540, "Phospholipase B-like")]},
]
OUTPUT_FILENAMES = {
    "ATP6V1A": "fig_03_07_atp6v1a_dense.png",
    "ATP6V1B2": "fig_03_08_atp6v1b2_dense.png",
    "H2BC11": "fig_03_09_h2bc11_dense.png",
    "PLBD1": "fig_03_10_plbd1_dense.png",
}


def load_sequences() -> dict[str, str]:
    with PIC2_INTEGRATED_JSON.open() as f:
        entries = json.load(f)
    sequences: dict[str, str] = {}
    for entry in entries:
        gene = entry.get("gene_name") or entry.get("gene") or entry.get("symbol")
        sequence = entry.get("protein_sequence")
        if gene and sequence and gene not in sequences:
            sequences[gene] = sequence
    return sequences


def build_model(device: torch.device) -> tuple[PIC, dict]:
    config_path = PIC2_MODEL_DIR / "model_config.json"
    model_path = PIC2_MODEL_DIR / "PIC_human_model.pth"
    with config_path.open() as f:
        config = json.load(f)

    model = PIC(
        input_shape=config["input_size"],
        hidden_units=config["hidden_size"],
        device=device,
        linear_drop=config["linear_drop"],
        attn_drop=config["attn_drop"],
        output_shape=config["output_size"],
        model_variant=config["model_variant"],
        num_heads=config["num_heads"],
    )

    state_dict = torch.load(model_path, map_location=device)
    patched_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("multihead_attention") or key.startswith("layerNorm"):
            patched_state_dict[f"backbone.{key.replace('layerNorm', 'layer_norm')}"] = value
        else:
            patched_state_dict[key] = value

    model.load_state_dict(patched_state_dict)
    model.to(device)
    model.eval()
    config["feature_dir"] = str(PIC2_SEQ_EMBED_DIR)
    return model, config


def annotate_top_pairs(ax, attn_matrix: np.ndarray, sequence: str, display_len: int) -> None:
    flat_idx = np.argsort(attn_matrix.ravel())[::-1]
    seen_pairs: list[tuple[int, int]] = []
    for flat in flat_idx:
        row_i = flat // display_len
        col_j = flat % display_len
        if row_i == col_j or (col_j, row_i) in seen_pairs:
            continue
        seen_pairs.append((row_i, col_j))
        if len(seen_pairs) >= 5:
            break

    for row_i, col_j in seen_pairs:
        aa_q = sequence[row_i] if row_i < len(sequence) else "?"
        aa_k = sequence[col_j] if col_j < len(sequence) else "?"
        label = f"{aa_q}{row_i + 1}\u2194{aa_k}{col_j + 1}"
        ax.plot(
            col_j + 0.5,
            row_i + 0.5,
            "o",
            color="#00FF99",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )
        offset_x = 6 if col_j < display_len * 0.75 else -6
        ha_val = "left" if offset_x > 0 else "right"
        ax.annotate(
            label,
            xy=(col_j + 0.5, row_i + 0.5),
            xytext=(offset_x, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="white",
            ha=ha_val,
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"),
        )


def render_dense_plot(gene_name: str, sequence: str, attn_matrix: np.ndarray, attn_1d: np.ndarray, domains: list[tuple[int, int, str]], output_dir: Path) -> None:
    display_len = min(len(sequence), attn_matrix.shape[0])
    tick_step = 100

    fig = plt.figure(figsize=(FIGSIZE["attention"][0], FIGSIZE["attention"][0]))
    gs = GridSpec(4, 1, height_ratios=[5.5, 2, 0.6, 0.6], hspace=0.35)

    ax_heat = fig.add_subplot(gs[0])
    sns.heatmap(
        attn_matrix,
        cmap=CMAPS["attention"],
        ax=ax_heat,
        cbar_kws={"label": "Attention Weight", "shrink": 0.8},
    )
    ax_heat.set_title(f"Attention Map: {gene_name}", fontsize=20, fontweight="bold", pad=15)
    ax_heat.set_ylabel("Query Residue", fontsize=13, fontweight="bold")
    ax_heat.set_xlabel("Key Residue", fontsize=13, fontweight="bold")
    tick_positions = np.arange(0, display_len, tick_step)
    tick_labels = (tick_positions + 1).astype(int)
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels(tick_labels, fontsize=11, rotation=0)
    ax_heat.set_yticks(tick_positions)
    ax_heat.set_yticklabels(tick_labels, fontsize=11, rotation=0)
    ax_heat.tick_params(axis="both", width=2, length=5)
    annotate_top_pairs(ax_heat, attn_matrix, sequence, display_len)
    cbar = ax_heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11, width=2)
    cbar.set_label("Attention Weight", fontsize=12, fontweight="bold")

    ax_1d = fig.add_subplot(gs[1])
    positions = np.arange(1, display_len + 1)
    ax_1d.plot(positions, attn_1d, color=NATURE_COLORS["dark_blue"], linewidth=2.0, label="Attention Prominence")
    ax_1d.fill_between(positions, 0, attn_1d, color=NATURE_COLORS["primary_blue"], alpha=0.2)
    peak_indices = sorted(np.argsort(attn_1d)[-5:])
    for peak_i, idx in enumerate(peak_indices):
        aa = sequence[idx] if idx < len(sequence) else "?"
        pos = idx + 1
        offset = 25 if peak_i % 2 == 0 else 55
        ax_1d.annotate(
            f"{aa}{pos}",
            xy=(pos, attn_1d[idx]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=NATURE_COLORS["dark_red"],
            arrowprops=dict(arrowstyle="->", lw=1, color=NATURE_COLORS["dark_red"]),
        )

    domain_colors = [
        NATURE_COLORS["primary_green"],
        NATURE_COLORS["primary_orange"],
        NATURE_COLORS["primary_purple"],
    ]
    for domain_i, (start, end, name) in enumerate(domains):
        if start <= display_len:
            end_clip = min(end, display_len)
            ax_1d.axvspan(start, end_clip, color=domain_colors[domain_i % 3], alpha=0.25)
            ax_1d.text(
                (start + end_clip) / 2,
                0.05,
                name,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    ax_1d.set_xlim(0.5, display_len + 0.5)
    ax_1d.set_ylim(0, 1.5)
    ax_1d.set_ylabel("Normalized\nImportance", fontsize=13, fontweight="bold")
    ax_1d_ticks = np.arange(0, display_len, tick_step)
    ax_1d.set_xticks(ax_1d_ticks + 1)
    ax_1d.set_xticklabels((ax_1d_ticks + 1).astype(int), fontsize=11)
    ax_1d.tick_params(axis="y", labelsize=12, width=2, length=5)
    for spine in ax_1d.spines.values():
        spine.set_linewidth(1.5)
    ax_1d.grid(alpha=0.3, linestyle="--")

    ax_hydro = fig.add_subplot(gs[2])
    hydro_vals = [HYDROPHOBICITY.get(aa, 0) for aa in sequence[:display_len]]
    hydro_img = np.array(hydro_vals).reshape(1, -1)
    im_hydro = ax_hydro.imshow(hydro_img, aspect="auto", cmap="RdYlBu_r", vmin=-4.5, vmax=4.5)
    ax_hydro.set_yticks([])
    ax_hydro.set_xticks([])
    ax_hydro.set_ylabel("Hydropho-\nbicity", fontsize=11, fontweight="bold", rotation=0, labelpad=48, va="center")
    for spine in ax_hydro.spines.values():
        spine.set_linewidth(1.5)
    cbar_hydro = plt.colorbar(im_hydro, ax=ax_hydro, orientation="vertical", fraction=0.015, pad=0.01)
    cbar_hydro.set_ticks([-4.5, 0, 4.5])
    cbar_hydro.set_ticklabels(["−4.5", "0", "+4.5"], fontsize=9)
    cbar_hydro.ax.tick_params(width=1.5, length=3)
    cbar_hydro.ax.set_title("Hydrophilic ← → Hydrophobic", fontsize=8, pad=4)

    ax_charge = fig.add_subplot(gs[3])
    charge_vals = [CHARGE.get(aa, 0) for aa in sequence[:display_len]]
    charge_img = np.array(charge_vals).reshape(1, -1)
    im_charge = ax_charge.imshow(charge_img, aspect="auto", cmap="PiYG", vmin=-1.5, vmax=1.5)
    ax_charge.set_yticks([])
    ax_charge.set_ylabel("Charge\n(pH 7)", fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")
    ax_charge.set_xlabel("Amino Acid Position", fontsize=13, fontweight="bold")
    ax_charge.tick_params(axis="x", labelsize=12, width=2, length=5)
    ax_charge_ticks = np.arange(0, display_len, tick_step)
    ax_charge.set_xticks(ax_charge_ticks)
    ax_charge.set_xticklabels((ax_charge_ticks + 1).astype(int), fontsize=11)
    for spine in ax_charge.spines.values():
        spine.set_linewidth(1.5)
    cbar_charge = plt.colorbar(im_charge, ax=ax_charge, orientation="vertical", fraction=0.015, pad=0.01)
    cbar_charge.set_ticks([-1, 0, 1])
    cbar_charge.set_ticklabels(["−1", "0", "+1"], fontsize=9)
    cbar_charge.ax.tick_params(width=1.5, length=3)
    cbar_charge.ax.set_title("(−) ← → (+)", fontsize=8, pad=4)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / OUTPUT_FILENAMES[gene_name]
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Generated dense visualization for {gene_name} to {save_path}")


def visualize_attention_dense() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config = build_model(device)
    sequences = load_sequences()
    dataset = PIC_Dataset(
        indexes=[protein["index"] for protein in PROTEINS],
        feature_dir=config["feature_dir"],
        label_name=config["label_name"],
        max_length=config["max_length"],
        feature_length=config["feature_length"],
        device=device,
    )

    for i, protein in enumerate(PROTEINS):
        gene_name = protein["gene"]
        sequence = sequences.get(gene_name)
        if not sequence:
            print(f"Skipping {gene_name}, sequence not found in integrated JSON")
            continue

        feature, _, start_padding_idx = dataset[i]
        feature = feature.unsqueeze(0)
        start_padding_idx = start_padding_idx.unsqueeze(0)
        with torch.no_grad():
            _, attn_weights = model(feature, start_padding_idx, get_attention=True)

        display_len = min(len(sequence), config["max_length"])
        attn_matrix = attn_weights[0, :display_len, :display_len].cpu().numpy()
        attn_1d = attn_matrix.mean(axis=0)
        attn_1d = (attn_1d - attn_1d.min()) / (attn_1d.max() - attn_1d.min() + 1e-8)
        render_dense_plot(gene_name, sequence, attn_matrix, attn_1d, protein["domains"], FIGURES_DIR)


if __name__ == "__main__":
    visualize_attention_dense()
