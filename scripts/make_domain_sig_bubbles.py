#!/usr/bin/env python3
"""Generate domain enrichment bubble plots for neutrophil protein groups."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Publication style tweaks
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = [
    'Arial',
    'DejaVu Sans',
    'Liberation Sans',
    'Bitstream Vera Sans',
    'sans-serif',
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

SRC = 'analysis/results/domain_enrichment_results.csv'
OUT = 'result/neutrophil_analysis/visualizations/B_domain_sig_bubbles_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PALETTE = {
    'primary': '#254E7B',
    'secondary': '#F08C2E',
    'tertiary': '#6AC1B8',
}
BUBBLE_CMAP = LinearSegmentedColormap.from_list(
    'domain_sig',
    ['#EFF3FB', PALETTE['tertiary'], PALETTE['secondary']],
)

GROUP_ORDER = [
    'Commonly Essential',
    'Human-Specific Essential',
    'Immune-Specific Essential',
]


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=12)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        try:
            label.set_fontweight('bold')
        except Exception:
            pass


def shorten(text: str, maxlen: int = 36) -> str:
    if not isinstance(text, str):
        return text
    return text if len(text) <= maxlen else text[: maxlen - 3] + '...'


def abbreviate_domain(s: str) -> str:
    if not isinstance(s, str):
        return s
    replacements = [
        ('Manganese/iron superoxide dismutase N-terminal', 'Mn/Fe SOD N-term'),
        ('Manganese', 'Mn'),
        ('Superoxide dismutase', 'SOD'),
    ]
    t = s
    for old, new in replacements:
        t = t.replace(old, new)
    return t


def prep_group(df: pd.DataFrame, group: str, topn: int = 18) -> pd.DataFrame:
    sub = df[df['group'] == group].copy()
    if sub.empty:
        return sub
    sub = sub[sub['count'] > 0].copy()
    if sub.empty:
        return sub
    sub['neglogq'] = -np.log10(sub['corrected_p_value'].replace(0, np.nextafter(0, 1)))
    sig = sub[sub['corrected_p_value'] < 0.05].copy()
    if sig.empty:
        sig = sub.sort_values(
            ['neglogq', 'fold_enrichment', 'percentage'],
            ascending=[False, False, False],
        ).head(topn)
    else:
        sig = sig.sort_values(
            ['neglogq', 'fold_enrichment', 'percentage'],
            ascending=[False, False, False],
        ).head(topn)
    sig['label'] = sig['domain'].apply(abbreviate_domain).apply(lambda s: shorten(s, 32))
    return sig


def plot_panel(ax, data: pd.DataFrame, title: str, vmin: float, vmax: float):
    if data.empty:
        ax.text(
            0.5,
            0.5,
            'No significant domains',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
        )
        ax.axis('off')
        return None
    data = data.sort_values('neglogq', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(data))[::-1]
    sizes = data['percentage'] * 48 + 140
    colors = data['neglogq']
    sc = ax.scatter(
        data['fold_enrichment'],
        y_pos,
        s=sizes,
        c=colors,
        cmap=BUBBLE_CMAP,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.92,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(range(len(data)))
    labels = list(data['label'])[::-1]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Fold Enrichment', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title(title, fontweight='bold')
    ax.axvline(1.0, ls='--', c=PALETTE['primary'], lw=1.2, alpha=0.6)
    style_ax(ax)
    ax.tick_params(axis='y', labelsize=9)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(9)
    return sc


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f'Missing enrichment data: {SRC}')
    df = pd.read_csv(SRC)
    if df.empty:
        raise ValueError(f'No data in {SRC}')

    groups = []
    for group in GROUP_ORDER:
        if group in df['group'].unique():
            groups.append(group)
    for group in df['group'].unique():
        if group not in groups:
            groups.append(group)

    prepared = []
    for group in groups:
        prepared.append(prep_group(df, group))

    neglog_vals = np.concatenate(
        [sub['neglogq'].values for sub in prepared if not sub.empty]
    ) if any(not sub.empty for sub in prepared) else np.array([])
    if neglog_vals.size > 0:
        vmax = float(np.nanpercentile(neglog_vals, 95))
        vmax = min(12.0, vmax) if np.isfinite(vmax) else 8.0
    else:
        vmax = 5.0
    vmin = 0.0

    fig, axes = plt.subplots(len(groups), 1, figsize=(9.0, 4.8 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    scatter_handles = []
    for idx, (ax, group, data) in enumerate(zip(axes, groups, prepared)):
        title = f'{group}: Domains'
        sc = plot_panel(ax, data, title, vmin=vmin, vmax=vmax)
        if sc is not None:
            scatter_handles.append(sc)
        if idx < len(letters):
            panel_label = letters[idx]
        else:
            panel_label = f'P{idx + 1}'
        ax.text(
            0.0,
            1.02,
            panel_label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            ha='left',
            va='bottom',
        )

    if scatter_handles:
        sc = scatter_handles[-1]
        fig.subplots_adjust(left=0.32, right=0.88, hspace=0.32)
        fig.canvas.draw()
        first_pos = axes[0].get_position()
        last_pos = axes[-1].get_position()
        y0 = last_pos.y0
        height = first_pos.y1 - last_pos.y0
        cax = fig.add_axes([0.90, y0, 0.02, height])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('-log10(q)', fontsize=13, fontweight='bold')
        cbar.outline.set_linewidth(1.5)
        cbar.ax.tick_params(labelsize=12, width=1.5)
    else:
        fig.subplots_adjust(left=0.32, right=0.95, hspace=0.28)

    fig.suptitle(
        'Significant Protein Domain Enrichment Across Neutrophil Groups',
        y=0.995,
        fontsize=15,
        fontweight='bold',
    )
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
