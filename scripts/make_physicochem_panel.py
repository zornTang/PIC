#!/usr/bin/env python3
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style (top-journal-like): thicker axes and ticks
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

PALETTE = {
    'primary': '#254E7B',
    'secondary': '#F08C2E',
    'tertiary': '#6AC1B8',
}

DATA_FOUR = 'analysis/data/neutrophil_four_group_classification.csv'
DATA_HUMAN = 'analysis/data/neutrophil_proteins_human_predictions.csv'
OUT = 'result/neutrophil_analysis/visualizations/C_physicochemical_panel_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

AA_MW = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
    'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

AA_GRAVY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

AA_CHARGE = {'R': 1, 'K': 1, 'D': -1, 'E': -1}


def clean_seq(s):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(s).upper())


def compute_props(seq: str):
    s = clean_seq(seq)
    if not s:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    L = len(s)
    counts = Counter(s)
    # Molecular weight (Da)
    mw = sum(AA_MW.get(a, 0) * n for a, n in counts.items())
    # GRAVY (Kyte–Doolittle average)
    gravy = sum(AA_GRAVY.get(a, 0) * n for a, n in counts.items()) / L
    # Net charge (R,K) - (D,E)
    net_charge = sum(AA_CHARGE.get(a, 0) * n for a, n in counts.items())
    # Charged AA percentage (% of R,K,D,E)
    charged_pct = (counts.get('R',0) + counts.get('K',0) + counts.get('D',0) + counts.get('E',0)) / L * 100.0
    # Hydrophobic AA percentage (% of A, I, L, M, F, W, Y, V)
    hydrophobic_pct = sum(counts.get(a, 0) for a in ['A','I','L','M','F','W','Y','V']) / L * 100.0
    # Approximate pI around 7 adjusted by charge density
    pos = counts.get('R', 0) + counts.get('K', 0)
    neg = counts.get('D', 0) + counts.get('E', 0)
    if L > 0:
        pi = 7.0 + (pos - neg) / L * 5.0
    else:
        pi = np.nan
    # Clamp pI to a reasonable range
    if not np.isnan(pi):
        pi = max(2.5, min(12.5, pi))
    return mw, pi, net_charge, gravy, charged_pct, hydrophobic_pct


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
    seq_map = df_h.set_index('protein_id')['sequence']

    df = df_four.copy()
    # Recompute groups using threshold=0.8 instead of relying on precomputed group
    th = 0.8
    if 'human_prob' in df.columns and 'immune_prob' in df.columns:
        h_ess = pd.to_numeric(df['human_prob'], errors='coerce') >= th
        i_ess = pd.to_numeric(df['immune_prob'], errors='coerce') >= th
        new_group = np.select(
            [h_ess & i_ess, h_ess & ~i_ess, ~h_ess & i_ess, ~h_ess & ~i_ess],
            ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential', 'Commonly Non-essential'],
            default='Commonly Non-essential'
        )
        df['group'] = new_group
    df['sequence'] = df['protein_id'].map(seq_map)

    props = df['sequence'].apply(lambda x: compute_props(x) if pd.notna(x) else (np.nan,)*6)
    df[['mw', 'pI', 'net_charge', 'gravy', 'charged_pct', 'hydrophobic_pct']] = pd.DataFrame(props.tolist(), index=df.index)

    # Plot only three groups (exclude Commonly Non-essential) with threshold-based grouping
    order = ['Human-Specific Essential', 'Immune-Specific Essential', 'Commonly Essential']
    palette = [PALETTE['primary'], PALETTE['secondary'], PALETTE['tertiary']]
    df['group'] = pd.Categorical(df['group'], categories=order, ordered=True)

    # use generous figure width to prevent x-axis label collisions
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    panels = [
        ('mw', 'Molecular Weight (kDa)', lambda y: y/1000.0),
        ('pI', 'Isoelectric Point (pI)', None),
        ('net_charge', 'Net Charge', None),
        ('hydrophobic_pct', 'Hydrophobic AA (%)', None),
    ]

    for ax, (col, title, transform) in zip(axes.flat, panels):
        data = [df[df['group'] == g][col].dropna().values for g in order]
        if transform is not None:
            data = [transform(v) for v in data]
        bp = ax.boxplot(data, patch_artist=True, labels=[g.replace(' ', '\n') for g in order])
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontweight='bold')
        # emphasize axis label font where applicable
        ax.set_xlabel('', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        style_ax(ax)
        formatted_labels = [g.replace(' ', '\n') for g in order]
        ax.set_xticklabels(formatted_labels, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', pad=14)

    fig.suptitle('Physicochemical Properties by Group', y=0.98, fontsize=13, fontweight='bold')
    # widen spacing between subplots for readability
    # revert to original inter-panel spacing but keep ample bottom margin
    plt.subplots_adjust(hspace=0.58, wspace=0.25, bottom=0.22, top=0.94)
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
