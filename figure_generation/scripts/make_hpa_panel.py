#!/usr/bin/env python3
"""
HPA blood-cell expression heatmap in unified thesis style.

Loads cached proteinatlas.tsv, draws a single-panel heatmap for
ATP6V1B2, ATP6V1A, PLBD1, H2BC11 across blood cell types.

Output: mythesis/figures/hpa_expression_heatmap.pdf / .png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from matplotlib.colors import LinearSegmentedColormap
from _fg_paths import FIGURES_NPJ_DIR, PIC2_ANALYSIS_DATA
from style_config import apply_style, PALETTE

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TSV_PATH   = PIC2_ANALYSIS_DATA / "proteinatlas.tsv"
NPJ_OUT    = FIGURES_NPJ_DIR

TARGET_GENES = ["ATP6V1B2", "ATP6V1A", "PLBD1", "H2BC11"]

SC_COL = "RNA single cell type specific nCPM"

# Blood / immune cell types to include (subset from pan-tissue SC column)
BLOOD_TYPES = [
    "Neutrophils", "Neutrophil progenitors",
    "Monocyte progenitors", "monocytes",
    "Kupffer cells",
    "Platelets",
]

CELL_LABELS = {
    "Neutrophils":            "Neutrophils",
    "Neutrophil progenitors": "Neutrophil prog.",
    "Monocyte progenitors":   "Monocyte prog.",
    "monocytes":              "Monocytes",
    "Kupffer cells":          "Kupffer cells",
    "Platelets":              "Platelets",
}

HEAT_BLUE_MATCHED = LinearSegmentedColormap.from_list(
    "heat_blue_matched",
    ["#F6F9FE", "#DCE7F7", "#B6CBEA", "#7EA3D3", "#4F79BB", PALETTE["human"]],
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_kv(value) -> dict:
    if not value or (isinstance(value, float) and np.isnan(value)):
        return {}
    result = {}
    for item in str(value).split(";"):
        item = item.strip()
        if ":" in item:
            ct, v = item.rsplit(":", 1)
            try:
                result[ct.strip()] = float(v.strip())
            except ValueError:
                pass
    return result


def load_sc_matrix(tsv_path: Path) -> pd.DataFrame:
    print(f"[INFO] Reading TSV: {tsv_path}")
    df_raw = pd.read_csv(tsv_path, sep="\t")
    df_g   = df_raw[df_raw["Gene"].isin(TARGET_GENES)].set_index("Gene")
    found  = [g for g in TARGET_GENES if g in df_g.index]

    records = {}
    for gene in found:
        kv = _parse_kv(df_g.loc[gene, SC_COL] if SC_COL in df_g.columns else "")
        records[gene] = {ct: kv.get(ct, 0.0) for ct in BLOOD_TYPES}
    df = pd.DataFrame(records).T
    # drop columns that are all-zero
    df = df.loc[:, (df > 0).any(axis=0)]
    print(f"[INFO] Matrix shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    gene_order = [g for g in TARGET_GENES if g in df.index]
    cols       = [c for c in BLOOD_TYPES if c in df.columns]
    df_plot    = df.loc[gene_order, cols]
    disp_cols  = [CELL_LABELS.get(c, c) for c in cols]

    n_genes = len(gene_order)
    n_cells = len(cols)

    # Neutrophil column index (exact Neutrophils, not progenitors)
    neut_idx = next(
        (i for i, c in enumerate(cols)
         if c.lower() == "neutrophils"),
        None,
    )

    fig, ax = plt.subplots(figsize=(max(8.6, n_cells * 1.35), max(4.0, n_genes * 1.28)))

    data = df_plot.values.astype(float)
    im   = ax.imshow(data, aspect="auto", cmap=HEAT_BLUE_MATCHED,
                     interpolation="nearest")

    # cell value annotations
    vmax = data.max() if data.max() > 0 else 1.0
    for i in range(n_genes):
        for j in range(n_cells):
            val = data[i, j]
            tc  = "white" if val > vmax * 0.62 else "#333333"
            ax.text(j, i, f"{int(val)}" if val > 0 else "—",
                    ha="center", va="center",
                    fontsize=10.5, color=tc, fontweight="bold")

    # highlight neutrophil column
    if neut_idx is not None:
        for i in range(n_genes):
            rect = plt.Rectangle(
                (neut_idx - 0.5, i - 0.5), 1, 1,
                linewidth=2.6, edgecolor=PALETTE["immune"],
                facecolor="none", zorder=5)
            ax.add_patch(rect)

    ax.set_xticks(range(n_cells))
    ax.set_xticklabels(disp_cols, rotation=35, ha="right", fontsize=12)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_order, fontsize=12.5, fontweight="bold")
    ax.tick_params(axis='x', width=2.0, length=5)
    ax.tick_params(axis='y', width=2.0, length=5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("nCPM  (HPA single-cell)", fontsize=11.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=1.8, length=4)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')
    cbar.outline.set_linewidth(1.8)

    ax.set_xlabel("Blood Cell Type", fontsize=12.5, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color('black')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df  = load_sc_matrix(TSV_PATH)
    fig = plot_heatmap(df)

    NPJ_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_OUT / "fig_18_hpa_expression_heatmap.png", dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
    plt.close(fig)
    print(f"Saved {NPJ_OUT / 'fig_18_hpa_expression_heatmap.png'}")


if __name__ == "__main__":
    main()
