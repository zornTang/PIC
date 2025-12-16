#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

NATURE = {
    'blue': '#4DBBD5',
    'red': '#E64B35',
    'green': '#00A087',
    'orange': '#F39B7F',
    'purple': '#8491B4',
    'gray': '#7F7F7F',
}

SRC = 'analysis/results/keyword_enrichment_results.csv'
OUT = 'result/neutrophil_analysis/visualizations/B_functional_enrichment_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Patterns to capture user's narrative
PATTERNS = [
    ('Cell cycle', re.compile(r'cell cycle|cell division|dna replication', re.I)),
    ('Homodimerization', re.compile(r'homodimer|dimer', re.I)),
    ('Migration/Motility', re.compile(r'migration|locomotion|chemotaxis', re.I)),
    ('Mechanical stimulus', re.compile(r'mechanical', re.I)),
    ('Lipid metabolism', re.compile(r'lipid|fatty acid|sterol', re.I)),
]


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=10)


def main():
    if not os.path.exists(SRC):
        print('No keyword_enrichment_results.csv found; aborting')
        return
    df = pd.read_csv(SRC)
    # keep best match for each (group, pattern)
    rows = []
    for label, pattern in PATTERNS:
        sub = df[df['keyword'].str.contains(pattern)] if 'keyword' in df.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        # for each group, take the most significant (lowest corrected p)
        for g, gsub in sub.groupby('group'):
            gsub = gsub.sort_values('corrected_p_value', ascending=True)
            row = gsub.iloc[0].copy()
            row['pattern_label'] = label
            rows.append(row)

    if not rows:
        print('No matching keywords for focus patterns; aborting minimal plot')
        # Still create an empty placeholder figure to avoid breaking caller
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No matching keywords found in enrichment results', ha='center', va='center')
        ax.axis('off')
        fig.savefig(OUT, bbox_inches='tight')
        return

    plot_df = pd.DataFrame(rows)
    # bubble encodings
    plot_df['neglogq'] = -np.log10(plot_df['corrected_p_value'].replace(0, np.nextafter(0, 1)))

    groups = ['Human-Specific Essential', 'Immune-Specific Essential']
    plot_df = plot_df[plot_df['group'].isin(groups)]
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=groups, ordered=True)

    ycats = [lab for lab, _ in PATTERNS]
    plot_df['pattern_label'] = pd.Categorical(plot_df['pattern_label'], categories=ycats, ordered=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x = plot_df['group'].cat.codes
    y = plot_df['pattern_label'].cat.codes
    sizes = plot_df['percentage'] * 15 + 10
    colors = plot_df['neglogq']
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap='RdYlBu_r', edgecolors='k', linewidths=0.3, alpha=0.9)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15)
    ax.set_yticks(range(len(ycats)))
    ax.set_yticklabels(ycats)
    ax.set_xlabel('Group')
    ax.set_ylabel('Functional theme (keyword proxy)')
    ax.set_title('Focused Functional Enrichment (Keyword proxies)')
    style_ax(ax)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('-log10(q)')
    cbar.outline.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()

