#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import textwrap
import re

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

SRC = 'analysis/results/subcellular_localization_distribution.csv'
OUT = 'result/neutrophil_analysis/visualizations/E_subcellular_localization_tj.png'
OUT_TOP = 'result/neutrophil_analysis/visualizations/E_subcellular_localization_top_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PALETTE = {
    'primary': '#254E7B',
    'secondary': '#F08C2E',
    'tertiary': '#6AC1B8',
}
HEAT_CMAP = LinearSegmentedColormap.from_list(
    'subcellular',
    ['#EFF3FB', PALETTE['primary']]
)

GROUPS = ['Human-Specific Essential', 'Immune-Specific Essential']
SHORT = {
    'Human-Specific Essential': 'human',
    'Immune-Specific Essential': 'immune',
}


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=12)
    for lbl in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
        try:
            lbl.set_fontweight('bold')
        except Exception:
            pass


def wrap_label(s: str, width=18):
    try:
        return '\n'.join(textwrap.wrap(s, width=width))
    except Exception:
        return s


def shorten_loc(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name
    # Common shortenings to keep meaning while reducing length
    s = s.replace('Mitochondrion', 'Mito')
    s = s.replace('Endoplasmic reticulum', 'ER')
    s = s.replace('Golgi apparatus', 'Golgi')
    s = s.replace('Extracellular region', 'Extracellular')
    s = s.replace('Extracellular space', 'Extracellular')
    s = s.replace('Plasma membrane', 'PM')
    s = s.replace('Cell membrane', 'PM')
    s = s.replace('Cytoplasm', 'Cytoplasm')
    s = s.replace('Nucleus; Cytoplasm', 'Nucleus/Cytoplasm')
    s = s.replace('Cytoplasm; Nucleus', 'Cytoplasm/Nucleus')
    s = s.replace('Nucleus; Chromosome', 'Nucleus/Chromosome')
    s = s.replace('Mitochondrion matrix', 'Mito matrix')
    s = s.replace('Peroxisome', 'Peroxi')
    s = s.replace('Lysosome', 'Lyso')
    s = s.replace('Endosome', 'Endosome')
    # For combined locations, force a clean two-line split at '/'
    if '/' in s:
        s = s.replace('/', '/\n')
    # Final wrap for very long
    return wrap_label(s, width=16)


def main():
    if not os.path.exists(SRC):
        print(f'Missing source: {SRC}')
        return
    df = pd.read_csv(SRC)

    # filter to essential groups only
    df = df[df['group'].isin(GROUPS)].copy()

    # pivot to matrix of percentage (rows: group, cols: location)
    mat = df.pivot_table(index='group', columns='location', values='percentage', aggfunc='mean')
    # select columns to avoid clutter: keep locations with any group > 2%
    keep = (mat.max(axis=0) > 2.0)
    mat = mat.loc[GROUPS, keep]

    # sort columns by overall mean descending
    mat = mat.loc[:, mat.mean(axis=0).sort_values(ascending=False).index]

    # Transpose to draw vertically (locations on y-axis)
    matV = mat.T
    locs = [shorten_loc(c) for c in matV.index]

    # dynamic portrait figure size
    height = min(16, 4 + 0.30*len(locs))
    # make columns narrower by reducing figure width
    fig, ax = plt.subplots(figsize=(5.0, height))
    vmax = max(15, float(np.nanmax(matV.values)))
    im = ax.imshow(matV.values, aspect='auto', cmap=HEAT_CMAP, vmin=0, vmax=vmax)

    ax.set_yticks(range(len(locs)))
    ax.set_yticklabels(locs)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([SHORT.get(g, g) for g in GROUPS], rotation=0)
    ax.set_xlabel('Essential Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Subcellular Localization', fontsize=12, fontweight='bold')
    ax.set_title('Subcellular Localization (Essential Groups)', fontweight='bold')
    style_ax(ax)

    # annotate values (rounded)
    for i in range(matV.shape[0]):
        for j in range(matV.shape[1]):
            val = matV.iat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8,
                        color='black')

    # colorbar on the right with compact ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    vmax = im.get_clim()[1]
    ticks = np.linspace(0, vmax, num=6)
    ticks = [round(t, 1) for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_label('Percentage (%)')
    cbar.outline.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')

    # Also produce a filtered version: top-N locations per group (union), for key highlights
    TOP_N = 5
    selected = set()
    for g in GROUPS:
        if g in mat.index:
            sel_idx = mat.loc[g].sort_values(ascending=False).head(TOP_N).index
            selected.update(sel_idx)
    if selected:
        mat_top = mat.loc[:, [c for c in mat.columns if c in selected]]
        if not mat_top.empty:
            mat_top = mat_top.loc[:, mat_top.mean(axis=0).sort_values(ascending=False).index]
            # horizontal layout: groups on y-axis, localizations along x-axis
            matH = mat_top
            locs2 = [shorten_loc(c) for c in matH.columns]
            width2 = min(16, 4 + 0.45 * len(locs2))
            fig2, ax2 = plt.subplots(figsize=(width2, 4.6))
            vmax_top = max(10, float(np.nanmax(matH.values)))
            im2 = ax2.imshow(matH.values, aspect='auto', cmap=HEAT_CMAP, vmin=0, vmax=vmax_top)
            ax2.set_xticks(range(len(locs2)))
            ax2.set_xticklabels(locs2, rotation=35, ha='right')
            ax2.set_yticks(range(len(GROUPS)))
            ax2.set_yticklabels([SHORT.get(g, g) for g in GROUPS])
            ax2.set_ylabel('Essential Groups', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Top Subcellular Localizations', fontsize=12, fontweight='bold')
            ax2.set_title('Top Localizations — Comparative Heatmap', fontweight='bold')
            style_ax(ax2)
            ax2.text(
                0.0,
                1.02,
                'F',
                transform=ax2.transAxes,
                fontsize=16,
                fontweight='bold',
                ha='left',
                va='bottom',
            )
            for i in range(matH.shape[0]):
                for j in range(matH.shape[1]):
                    val = matH.iat[i, j]
                    if not np.isnan(val):
                        ax2.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9, color='black')
            fig2.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.32)
            cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
            vmax2 = im2.get_clim()[1]
            ticks2 = np.linspace(0, vmax2, num=5)
            ticks2 = [round(t, 1) for t in ticks2]
            cbar2.set_ticks(ticks2)
            cbar2.set_label('Percentage (%)')
            cbar2.outline.set_linewidth(1.5)
            fig2.savefig(OUT_TOP, bbox_inches='tight')
            print(f'Saved {OUT_TOP}')


if __name__ == '__main__':
    main()
