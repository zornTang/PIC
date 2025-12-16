#!/usr/bin/env python3
"""
Generate figures requested by the user:

A. Global model differences (scatter of human vs immune probabilities)
B. Biological function differences (reuse enrichment bubble plots)
C. Physicochemical and structural explanations (multi-violin/box)
D. Convergent inference focusing on PLBD1 (annotated summary figure)

Inputs (existing in repo):
- analysis/data/neutrophil_four_group_classification.csv
- analysis/data/neutrophil_proteins_human_predictions.csv
- analysis/data/neutrophil_immune_ensemble_predictions.csv
- analysis/results/{keyword_enrichment_bubble.png, domain_enrichment_bubble.png, subcellular_localization_heatmap.png}

Outputs:
- result/neutrophil_analysis/visualizations/A_overall_scatter.png
- result/neutrophil_analysis/visualizations/B_enrichment_bubble.png (copied)
- result/neutrophil_analysis/visualizations/C_physicochemical_violin.png
- result/neutrophil_analysis/visualizations/C_subcellular_localization.png (copied)
- result/neutrophil_analysis/visualizations/D_PLBD1_focus.png
"""

import os
import shutil
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Matplotlib defaults for consistent style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


NATURE = {
    'blue': '#4DBBD5',
    'red': '#E64B35',
    'green': '#00A087',
    'orange': '#F39B7F',
    'purple': '#8491B4',
    'gray': '#7F7F7F',
}


DATA_FOUR = 'analysis/data/neutrophil_four_group_classification.csv'
DATA_HUMAN = 'analysis/data/neutrophil_proteins_human_predictions.csv'
DATA_IMMUNE = 'analysis/data/neutrophil_immune_ensemble_predictions.csv'

RESULT_DIR = 'result/neutrophil_analysis/visualizations'
os.makedirs(RESULT_DIR, exist_ok=True)


def _extract_gene_name(protein_id: str) -> str:
    try:
        parts = [p.strip() for p in str(protein_id).split('|')]
        for p in parts:
            if p.strip().startswith('Gene:'):
                return p.split(':', 1)[1].strip()
    except Exception:
        pass
    return str(protein_id)


def load_data():
    df_four = pd.read_csv(DATA_FOUR)
    df_four['gene'] = df_four['protein_id'].apply(_extract_gene_name)

    # Read predictions (for sequences and PES)
    df_h = pd.read_csv(DATA_HUMAN)
    df_h['gene'] = df_h['protein_id'].apply(_extract_gene_name)
    df_i = pd.read_csv(DATA_IMMUNE)
    df_i['gene'] = df_i['protein_id'].apply(_extract_gene_name)

    return df_four, df_h, df_i


def plot_global_scatter(df_four: pd.DataFrame, outpath: str):
    # Scatter of human_prob vs immune_prob colored by group
    plt.figure(figsize=(7.2, 6))
    ax = plt.gca()
    groups = {
        'Commonly Non-essential': NATURE['gray'],
        'Human-Specific Essential': NATURE['blue'],
        'Immune-Specific Essential': NATURE['red'],
        'Commonly Essential': NATURE['green'],
    }

    for g, color in groups.items():
        sub = df_four[df_four['group'] == g]
        ax.scatter(sub['immune_prob'], sub['human_prob'], s=10, alpha=0.4, c=color, label=g, edgecolors='none')

    # Reference lines
    ax.plot([0, 1], [0, 1], ls='--', c='k', lw=1, alpha=0.5)
    ax.axhline(0.5, ls=':', c='gray', lw=1)
    ax.axvline(0.5, ls=':', c='gray', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Immune PES')
    ax.set_ylabel('Human PES')
    ax.set_title('Human vs Immune PES (Global Differences)')
    ax.legend(markerscale=2, frameon=False, loc='lower right')

    # Correlation
    r, p = stats.pearsonr(df_four['human_prob'], df_four['immune_prob'])
    ax.text(0.03, 0.97, f'Pearson r={r:.3f}\nP={p:.1e}', transform=ax.transAxes,
            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


AA_PROP = {
    'mw': {'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15, 'Q': 146.15, 'E': 147.13,
           'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19,
           'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15},
    'hydro': {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,'Q': -3.5, 'E': -3.5, 'G': -0.4,
              'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
              'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2},
    'charge': {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
               'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0},
}


def _clean_seq(s):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(s).upper())


def compute_physchem(seq: str):
    s = _clean_seq(seq)
    if not s:
        return np.nan, np.nan, np.nan, np.nan
    L = len(s)
    counts = Counter(s)
    mw = sum(AA_PROP['mw'].get(aa, 0) * n for aa, n in counts.items())
    hydro = sum(AA_PROP['hydro'].get(aa, 0) * n for aa, n in counts.items()) / L
    charge = sum(AA_PROP['charge'].get(aa, 0) * n for aa, n in counts.items())
    aromatic = sum(counts.get(a, 0) for a in ['F', 'W', 'Y']) / L * 100
    return L, mw, hydro, charge, aromatic


def plot_physicochemical(df_four: pd.DataFrame, df_h: pd.DataFrame, outpath: str):
    # Merge sequences onto four-group table via protein_id
    seq_map = df_h.set_index('protein_id')['sequence']
    df = df_four.copy()
    df['sequence'] = df['protein_id'].map(seq_map)

    props = df['sequence'].apply(lambda x: compute_physchem(x) if pd.notna(x) else (np.nan,)*5)
    df[['length', 'mw', 'hydro', 'charge', 'aromatic_pct']] = pd.DataFrame(props.tolist(), index=df.index)

    order = ['Commonly Non-essential', 'Human-Specific Essential', 'Immune-Specific Essential', 'Commonly Essential']
    df['group'] = pd.Categorical(df['group'], categories=order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    order = ['Commonly Non-essential', 'Human-Specific Essential', 'Immune-Specific Essential', 'Commonly Essential']
    palette = [NATURE['gray'], NATURE['blue'], NATURE['red'], NATURE['green']]

    def boxplot_property(ax, prop, title, ylabel):
        data = [df[df['group'] == g][prop].dropna().values for g in order]
        bp = ax.boxplot(data, patch_artist=True, labels=[g.replace(' ', '\n') for g in order])
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=0)

    boxplot_property(axes[0, 0], 'hydro', 'Hydrophobicity Index by Group', 'Hydrophobicity')
    boxplot_property(axes[0, 1], 'aromatic_pct', 'Aromatic AA (%) by Group', 'Aromatic AA (%)')
    boxplot_property(axes[1, 0], 'charge', 'Net Charge by Group', 'Net Charge')
    boxplot_property(axes[1, 1], 'length', 'Sequence Length by Group', 'Length (aa)')

    plt.suptitle('Physicochemical Profiles by Essentiality Group', y=1.02, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def copy_enrichment_and_localization():
    # Copy existing enrichment bubble and localization heatmap
    srcs = [
        ('analysis/results/keyword_enrichment_bubble.png', os.path.join(RESULT_DIR, 'B_enrichment_bubble.png')),
        ('analysis/results/subcellular_localization_heatmap.png', os.path.join(RESULT_DIR, 'C_subcellular_localization.png')),
    ]
    for src, dst in srcs:
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def plot_plbd1_focus(df_four: pd.DataFrame, df_h: pd.DataFrame, outpath: str):
    df = df_four.copy()
    df['gene'] = df['protein_id'].apply(_extract_gene_name)

    # Identify PLBD1 entries
    plbd1_rows = df[df['gene'].str.upper() == 'PLBD1']
    # Pick the one with highest immune_prob
    plbd1 = plbd1_rows.sort_values('immune_prob', ascending=False).head(1)

    # Merge sequence for physchem
    seq_map = df_h.set_index('protein_id')['sequence']
    if not plbd1.empty:
        seq = seq_map.get(plbd1.iloc[0]['protein_id'], np.nan)
        L, mw, hydro, charge, aromatic = compute_physchem(seq)
    else:
        L = mw = hydro = charge = aromatic = np.nan

    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: scatter with annotation
    ax = axes[0, 0]
    ax.scatter(df['immune_prob'], df['human_prob'], s=10, alpha=0.15, c=NATURE['gray'], edgecolors='none')
    # highlight immune-specific essential
    imm = df[df['group'] == 'Immune-Specific Essential']
    ax.scatter(imm['immune_prob'], imm['human_prob'], s=12, alpha=0.5, c=NATURE['red'], edgecolors='none', label='Immune-Specific Essential')
    # annotate PLBD1
    if not plbd1.empty:
        x, y = float(plbd1.iloc[0]['immune_prob']), float(plbd1.iloc[0]['human_prob'])
        ax.scatter([x], [y], s=70, c='black', marker='*', zorder=5, label='PLBD1')
        ax.annotate('PLBD1', (x, y), xytext=(10, -10), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.plot([0, 1], [0, 1], ls='--', c='k', lw=1, alpha=0.5)
    ax.axhline(0.5, ls=':', c='gray', lw=1)
    ax.axvline(0.5, ls=':', c='gray', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Immune PES')
    ax.set_ylabel('Human PES')
    ax.set_title('Model Outputs → Immune-specific Identification')
    ax.legend(frameon=False, loc='lower right')

    # Panel B: PLBD1 vs immune-specific distribution (hydrophobicity, length)
    ax2 = axes[0, 1]
    # compute group stats
    seq_map = df_h.set_index('protein_id')['sequence']
    imm['sequence'] = imm['protein_id'].map(seq_map)
    props = imm['sequence'].apply(lambda s: compute_physchem(s) if pd.notna(s) else (np.nan,)*5)
    imm[['length', 'mw', 'hydro', 'charge', 'aromatic_pct']] = pd.DataFrame(props.tolist(), index=imm.index)
    # single-group boxplot for hydrophobicity with PLBD1 overlay
    hvals = imm['hydro'].dropna().values
    if hvals.size > 0:
        bp = ax2.boxplot([hvals], patch_artist=True, labels=['Immune-Specific'])
        bp['boxes'][0].set_facecolor(NATURE['red'])
        bp['boxes'][0].set_alpha(0.7)
    if not plbd1.empty:
        ax2.scatter([0], [hydro], color='black', marker='*', s=80, zorder=5, label='PLBD1')
    ax2.set_title('Hydrophobicity (Immune-Specific)')
    ax2.set_xlabel('')
    ax2.legend(frameon=False, loc='lower right')

    # Panel C: PLBD1 basic sequence features
    ax3 = axes[1, 0]
    bars = ['Length', 'Hydrophobicity', 'Charge', 'Aromatic %']
    vals = [L, hydro, charge, aromatic]
    colors = [NATURE['blue'], NATURE['orange'], NATURE['purple'], NATURE['green']]
    ax3.bar(bars, vals, color=colors)
    ax3.set_title('PLBD1 Sequence-Derived Features')
    ax3.grid(True, axis='y', alpha=0.3)

    # Panel D: Flow summary schematic
    ax4 = axes[1, 1]
    ax4.axis('off')
    text = (
        'Model outputs → identify Immune-specific (PES≥0.8 immune, <0.8 human)\n'
        '↓\n'
        'Characterize molecular traits (hydrophobicity, length, charge, aromatic %)\n'
        '↓\n'
        'Biological significance supported by enrichment and localization\n'
        '↓\n'
        'Representative: PLBD1 (immune-specific essential)'
    )
    ax4.text(0.02, 0.9, text, va='top')

    plt.suptitle('Convergent Inference Centered on PLBD1', y=1.02, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def main():
    df_four, df_h, df_i = load_data()

    # A. Global scatter
    plot_global_scatter(df_four, os.path.join(RESULT_DIR, 'A_overall_scatter.png'))

    # B. Copy enrichment visuals and C. localization
    copy_enrichment_and_localization()

    # C. Physicochemical violins
    plot_physicochemical(df_four, df_h, os.path.join(RESULT_DIR, 'C_physicochemical_violin.png'))

    # D. PLBD1 focus
    plot_plbd1_focus(df_four, df_h, os.path.join(RESULT_DIR, 'D_PLBD1_focus.png'))

    print(f"Figures written to: {RESULT_DIR}")


if __name__ == '__main__':
    main()
