#!/usr/bin/env python3
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy import stats

# Top-journal style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

OUT = 'result/neutrophil_analysis/visualizations/C_aa_composition_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DATA_FOUR = 'analysis/data/neutrophil_four_group_classification.csv'
DATA_HUMAN = 'analysis/data/neutrophil_proteins_human_predictions.csv'

AA_LIST = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

PRIMARY = "#254E7B"
SECONDARY = "#F08C2E"
TERTIARY = "#6AC1B8"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "aa_heat",
    [
        (0.0, "#F0F5FF"),
        (0.20, "#D1DFF4"),
        (0.45, "#A6C4EA"),
        (0.70, "#6F98D1"),
        (0.88, "#3F67B1"),
        (1.0, PRIMARY),
    ],
)

def clean_seq(s):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(s).upper())


def aa_percentages(seq):
    s = clean_seq(seq)
    if not s:
        return {aa: np.nan for aa in AA_LIST}
    L = len(s)
    cnt = Counter(s)
    return {aa: cnt.get(aa, 0) / L * 100.0 for aa in AA_LIST}


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=12)
    for lbl in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
        try:
            lbl.set_fontweight('bold')
        except Exception:
            pass


def main():
    df_four = pd.read_csv(DATA_FOUR)
    df_h = pd.read_csv(DATA_HUMAN)

    # Threshold-based grouping at 0.8
    th = 0.8
    if {'human_prob','immune_prob'}.issubset(df_four.columns):
        h_ess = pd.to_numeric(df_four['human_prob'], errors='coerce') >= th
        i_ess = pd.to_numeric(df_four['immune_prob'], errors='coerce') >= th
        groups = np.select(
            [h_ess & i_ess, h_ess & ~i_ess, ~h_ess & i_ess, ~h_ess & ~i_ess],
            ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential', 'Commonly Non-essential'],
            default='Commonly Non-essential'
        )
        df_four = df_four.copy()
        df_four['group'] = groups

    # Map sequences
    seq_map = df_h.set_index('protein_id')['sequence']
    df_four['sequence'] = df_four['protein_id'].map(seq_map)

    # Only keep two groups (exclude Commonly Non-essential and Commonly Essential)
    order = ['Human-Specific Essential', 'Immune-Specific Essential']
    df_four = df_four[df_four['group'].isin(order)].copy()
    df_four['group'] = pd.Categorical(df_four['group'], categories=order, ordered=True)

    # Compute AA composition per protein
    comp = df_four['sequence'].apply(lambda s: aa_percentages(s) if pd.notna(s) else {aa: np.nan for aa in AA_LIST})
    comp_df = pd.DataFrame(list(comp.values), index=df_four.index)
    df = pd.concat([df_four[['group']], comp_df], axis=1)

    # Aggregate mean composition by group
    mean_comp = df.groupby('group')[AA_LIST].mean()

    # Heatmap
    fig, ax = plt.subplots(figsize=(11, 4.8))
    raw_max = float(np.nanmax(mean_comp.values))
    if not np.isfinite(raw_max) or raw_max <= 0:
        raw_max = 1.0
    vmax = min(10.0, raw_max * 1.05)
    if vmax < 1.0:
        vmax = 1.0
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=vmax)
    im = ax.imshow(
        mean_comp.values,
        aspect='auto',
        cmap=HEAT_CMAP,
        norm=norm,
    )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xticks(range(len(AA_LIST)))
    ax.set_xticklabels(AA_LIST)
    ax.set_xlabel('Amino Acid', fontsize=12, fontweight='bold')
    ax.set_ylabel('Group (PES ≥ 0.8 based)', fontsize=12, fontweight='bold')
    ax.set_title('Amino Acid Composition Comparison (%)', fontweight='bold')
    style_ax(ax)

    # Annotate values for clarity (sparse to avoid clutter)
    # Annotate all valid cells for clarity (was: only >=5%)
    for i in range(len(order)):
        for j in range(len(AA_LIST)):
            val = mean_comp.iat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                        color='black')

    # Highlight specified residue categories with palette accents
    try:
        human_idx = list(order).index('Human-Specific Essential')
        immune_idx = list(order).index('Immune-Specific Essential')

        # Unified red highlights for all specified residues
        highlight_aa = ['K','R','A','V','L','F','N','Q','P']
        for aa in highlight_aa:
            if aa not in mean_comp.columns:
                continue
            j = AA_LIST.index(aa)
            # double-stroke rectangles: outer white, inner red for strong contrast
            for i in [human_idx, immune_idx]:
                rect_outer = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    linewidth=3.0,
                    edgecolor='white',
                    zorder=4,
                )
                ax.add_patch(rect_outer)
                rect_inner = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    linewidth=2.2,
                    edgecolor=SECONDARY,
                    zorder=5,
                )
                ax.add_patch(rect_inner)
            # annotate delta on immune row (red font)
            delta = mean_comp.loc['Immune-Specific Essential', aa] - mean_comp.loc['Human-Specific Essential', aa]
            ax.text(
                j,
                immune_idx - 0.42,
                f"Δ{delta:+.1f}",
                ha='center',
                va='bottom',
                fontsize=7,
                color=SECONDARY,
                fontweight='bold',
                zorder=6,
            )
    except Exception:
        pass

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, round(vmax, 1)])
    cbar.set_label('Percentage (%)')
    cbar.outline.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
