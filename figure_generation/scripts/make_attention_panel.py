#!/usr/bin/env python3
"""
ATP6V1B2 attention weight + functional site figure in unified thesis style.

Loads ESM2 embeddings, runs the PIC model, extracts per-residue attention,
overlays UniProt functional annotations.  Caches the attention array as .npy
so subsequent re-plots skip the model inference step.

Output: mythesis/figures/atp6v1b2_attention_functional_sites.pdf / .png
        PIC/analysis/results/atp6v1b2_attn_norm.npy  (cache)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from _fg_paths import (
    CODE_ROOT,
    FIGURES_NPJ_DIR,
    PIC2_ANALYSIS_RESULTS,
    PIC2_INTEGRATED_JSON,
    PIC2_MODEL_DIR,
)
from style_config import apply_style, PALETTE, panel_label

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULT_DIR   = PIC2_ANALYSIS_RESULTS
PIC_CODE_DIR = CODE_ROOT / "code"
NPJ_OUT      = FIGURES_NPJ_DIR

PROTEIN_SEQ_JSON = PIC2_INTEGRATED_JSON
MODEL_PATH       = PIC2_MODEL_DIR / "PIC_human_model.pth"
MODEL_CONFIG     = PIC2_MODEL_DIR / "model_config.json"
ESM2_MODEL_PATH  = CODE_ROOT / "pretrained_model" / "esm2_t33_650M_UR50D.pt"
ATTN_CACHE       = RESULT_DIR / "atp6v1b2_attn_norm.npy"

UNIPROT_ID = "P21281"
GENE_NAME  = "ATP6V1B2"
MAX_LENGTH = 1000

# ---------------------------------------------------------------------------
# Functional site annotations (UniProt + curated)
# ---------------------------------------------------------------------------
FEATURES = [
    {"type": "Region",           "description": "V-ATPase B domain",
     "start": 76,  "end": 499},
    {"type": "Region",           "description": "Walker A (P-loop)",
     "start": 148, "end": 155},
    {"type": "Region",           "description": "Walker B",
     "start": 192, "end": 197},
    {"type": "Modified residue", "description": "Y68 (ABL1 phospho-site)",
     "start": 68,  "end": 68},
]

# Colors mapped to PALETTE
REGION_COLORS = [
    PALETTE["neutral"],        # V-ATPase B domain — light gray fill
    PALETTE["common"],         # Walker A           — teal fill
    PALETTE["common"],         # Walker B           — teal fill
]


# ---------------------------------------------------------------------------
# Sequence loader
# ---------------------------------------------------------------------------
def load_sequence(json_path: Path, gene: str) -> str:
    with open(json_path) as fh:
        entries = json.load(fh)
    if isinstance(entries, list):
        for entry in entries:
            if entry.get("gene_name") == gene:
                return entry["protein_sequence"]
        raise KeyError(f"{gene} not found in {json_path}")
    # dict format: {gene: sequence}
    return entries[gene]


# ---------------------------------------------------------------------------
# ESM2 + PIC attention extraction
# ---------------------------------------------------------------------------
def compute_attention(sequence: str) -> np.ndarray:
    import torch
    import esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ESM2
    print("[INFO] Loading ESM2 …")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(str(ESM2_MODEL_PATH))
    esm_model = esm_model.to(device).eval()

    seq = sequence[:MAX_LENGTH]
    L   = len(seq)
    bc  = alphabet.get_batch_converter()
    _, _, tokens = bc([(GENE_NAME, seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        reps = esm_model(tokens, repr_layers=[33])["representations"][33]
    token_embs = reps[:, 1:L + 1, :]

    embed_dim = token_embs.shape[-1]
    feature   = torch.zeros(1, MAX_LENGTH, embed_dim, dtype=torch.float32, device=device)
    feature[0, :L, :] = token_embs[0]
    del esm_model

    # PIC model
    print("[INFO] Loading PIC model …")
    sys.path.insert(0, str(PIC_CODE_DIR))
    from module.PIC import PIC  # type: ignore

    with open(MODEL_CONFIG) as fh:
        cfg = json.load(fh)

    model = PIC(
        input_shape=cfg.get("input_size", cfg.get("input_shape", 1280)),
        output_shape=cfg.get("output_size", cfg.get("output_shape", 1)),
        hidden_units=cfg.get("hidden_size", cfg.get("hidden_units", 320)),
        attn_drop=cfg.get("attn_drop", 0.0),
        linear_drop=cfg.get("linear_drop", 0.0),
        device=device,
        model_variant=cfg.get("model_variant", "attention"),
        num_heads=cfg.get("num_heads", 1),
    ).to(device)

    sd = torch.load(str(MODEL_PATH), map_location=device)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("multihead_attention") or k.startswith("layerNorm"):
            new_sd["backbone." + k.replace("layerNorm", "layer_norm")] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    spi = torch.tensor([L], dtype=torch.long, device=device)
    with torch.no_grad():
        _, attn_w = model(feature, spi, get_attention=True)

    attn_2d  = attn_w[0, :L, :L]
    attn_1d  = attn_2d.mean(dim=0).cpu().numpy()
    mn, mx   = attn_1d.min(), attn_1d.max()
    attn_norm = (attn_1d - mn) / (mx - mn + 1e-8)

    np.save(str(ATTN_CACHE), attn_norm)
    print(f"[INFO] Saved attention cache → {ATTN_CACHE}")
    return attn_norm


# ---------------------------------------------------------------------------
# Biophysical helpers
# ---------------------------------------------------------------------------
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHG = {"R": 1, "K": 1, "D": -1, "E": -1}


def _sw(arr: np.ndarray, w: int = 9) -> np.ndarray:
    h = w // 2
    p = np.pad(arr, h, mode="edge")
    return np.array([p[i:i + w].mean() for i in range(len(arr))])


def hydrophobicity(seq: str) -> np.ndarray:
    return _sw(np.array([KD.get(aa, 0.0) for aa in seq]))


def net_charge(seq: str) -> np.ndarray:
    return _sw(np.array([CHG.get(aa, 0.0) for aa in seq], dtype=float))


# ---------------------------------------------------------------------------
# Figure  (3 panels: attention | hydrophobicity | charge)
# ---------------------------------------------------------------------------
def plot_attention(attn_norm: np.ndarray, sequence: str) -> plt.Figure:
    from matplotlib.gridspec import GridSpec

    L    = len(attn_norm)
    pos  = np.arange(1, L + 1)
    seq  = sequence[:L]
    hydr = hydrophobicity(seq)
    chrg = net_charge(seq)

    fig = plt.figure(figsize=(12.2, 8.8), layout="constrained")
    gs  = GridSpec(5, 1, figure=fig, hspace=0.08,
                   height_ratios=[3, 3, 3, 2, 2])
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[4, 0], sharex=ax1)

    # ── Panel 1: attention curve + functional sites ──────────────────────────
    ax1.plot(pos, attn_norm, color=PALETTE["human"], lw=2.6, alpha=0.95,
             label="Attention weight", zorder=3)
    ax1.fill_between(pos, attn_norm, alpha=0.18, color=PALETTE["human"], zorder=2)
    ax1.axhline(0.5, color=PALETTE["ref_line"], ls=":", lw=1.8, zorder=2,
                label="Threshold (0.5)")

    # region spans
    region_handles = []
    rc_idx = 0
    for feat in FEATURES:
        if feat["type"] != "Region":
            continue
        col   = REGION_COLORS[rc_idx % len(REGION_COLORS)]
        label = feat["description"]
        for _ax in (ax1, ax2, ax3):
            _ax.axvspan(feat["start"], feat["end"],
                        alpha=0.12, color=col, zorder=1)
        region_handles.append(mpatches.Patch(facecolor=col, alpha=0.4,
                                              label=label, edgecolor="none"))
        rc_idx += 1

    # Walker A / B centre lines
    for feat in FEATURES:
        if feat["type"] == "Region" and feat["description"] in ("Walker A (P-loop)", "Walker B"):
            mid = (feat["start"] + feat["end"]) // 2
            ax1.axvline(mid, color=PALETTE["common"], ls="--", lw=1.6,
                        alpha=0.8, zorder=3)
            ax1.text(mid + 3, 1.05, feat["description"].split()[0],
                     fontsize=13.0, color=PALETTE["common"], va="top",
                     fontweight="bold")

    # Y68 line + annotation
    for feat in FEATURES:
        if feat["type"] == "Modified residue":
            p68      = feat["start"]
            attn_p68 = float(attn_norm[p68 - 1]) if p68 <= L else 0.0
            for _ax in (ax1, ax2, ax3):
                _ax.axvline(p68, color=PALETTE["immune"], ls="--", lw=2.2,
                            alpha=0.85, zorder=4)
            ax1.annotate(
                f"Y{p68}  (ATN={attn_p68:.3f})",
                xy=(p68, attn_p68),
                xytext=(p68 + 22, attn_p68 + 0.10),
                fontsize=14.0, color=PALETTE["immune"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["immune"],
                                lw=2.0, connectionstyle="arc3,rad=-0.25"),
                zorder=5,
            )

    ax1.set_ylim(-0.04, 1.18)
    ax1.set_ylabel("Norm. Attention", fontsize=16.5, fontweight="bold")
    ax1.tick_params(labelbottom=False)

    legend_elements = [
        plt.Line2D([0], [0], color=PALETTE["human"], lw=2.6,
                   label="Attention weight"),
        plt.Line2D([0], [0], color=PALETTE["ref_line"], lw=1.8, ls=":",
                   label="Threshold (0.5)"),
        plt.Line2D([0], [0], color=PALETTE["immune"], lw=2.2, ls="--",
                   label="Y68  ABL1 phospho-site"),
    ] + region_handles
    ax1.legend(handles=legend_elements, fontsize=13.5, loc="upper right",
               framealpha=0.9, edgecolor=PALETTE["neutral"], ncol=2)

    # ── Panel 2: hydrophobicity ──────────────────────────────────────────────
    ax2.plot(pos, hydr, color=PALETTE["common"], lw=2.2, alpha=0.95)
    ax2.fill_between(pos, hydr, 0, where=hydr > 0, interpolate=True,
                     color=PALETTE["immune"], alpha=0.38, label="Hydrophobic")
    ax2.fill_between(pos, hydr, 0, where=hydr <= 0, interpolate=True,
                     color=PALETTE["human"], alpha=0.28, label="Hydrophilic")
    ax2.axhline(0, color=PALETTE["ref_line"], lw=1.5, ls="--")
    ax2.set_ylabel("Hydrophobicity\n(KD)", fontsize=15.0, fontweight="bold")
    ax2.legend(fontsize=12.5, frameon=False, loc="upper right")
    ax2.tick_params(labelbottom=False)

    # ── Panel 3: net charge ──────────────────────────────────────────────────
    ax3.plot(pos, chrg, color=PALETTE["common"], lw=2.2, alpha=0.95)
    ax3.fill_between(pos, chrg, 0, where=chrg > 0, interpolate=True,
                     color=PALETTE["immune"], alpha=0.38, label="Positive")
    ax3.fill_between(pos, chrg, 0, where=chrg <= 0, interpolate=True,
                     color=PALETTE["human"], alpha=0.28, label="Negative")
    ax3.axhline(0, color=PALETTE["ref_line"], lw=1.5, ls="--")
    ax3.set_ylabel("Net Charge\n(w=9)", fontsize=15.0, fontweight="bold")
    ax3.set_xlabel("Amino Acid Position", fontsize=16.5, fontweight="bold")
    ax3.legend(fontsize=12.5, frameon=False, loc="upper right")

    ax1.set_xlim(1, L)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis="both", labelsize=17.0, width=2.2, length=5.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Always load sequence (needed for biophysical panels)
    sequence = load_sequence(PROTEIN_SEQ_JSON, GENE_NAME)

    # 2. Load attention (from cache or model)
    if ATTN_CACHE.exists():
        print(f"[INFO] Loading cached attention array from {ATTN_CACHE}")
        attn_norm = np.load(str(ATTN_CACHE))
    else:
        print("[INFO] Cache not found — running model inference …")
        attn_norm = compute_attention(sequence)

    print(f"[INFO] Attention profile length: {len(attn_norm)}")

    # 3. Plot
    fig = plot_attention(attn_norm, sequence)

    NPJ_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_OUT / "fig_06_atp6v1b2_attention_functional_sites.png", dpi=600, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Saved {NPJ_OUT / 'fig_06_atp6v1b2_attention_functional_sites.png'}")


if __name__ == "__main__":
    main()
