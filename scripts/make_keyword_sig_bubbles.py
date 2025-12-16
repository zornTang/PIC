#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

SRC = 'analysis/results/keyword_enrichment_results.csv'
OUT = 'result/neutrophil_analysis/visualizations/F_keyword_enrichment_bubbles_tj.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PALETTE = {
    'primary': '#254E7B',
    'secondary': '#F08C2E',
    'tertiary': '#6AC1B8',
}
BUBBLE_CMAP = LinearSegmentedColormap.from_list(
    'keyword_sig',
    ['#EFF3FB', PALETTE['tertiary'], PALETTE['secondary']]
)


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=12)
    for lbl in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
        try:
            lbl.set_fontweight('bold')
        except Exception:
            pass


def shorten(text: str, maxlen: int = 35) -> str:
    if not isinstance(text, str):
        return text
    return text if len(text) <= maxlen else text[:maxlen-3] + '...'


def abbreviate_keyword(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Common keyword shortenings while keeping meaning
    repl = [
        ('Alternative splicing', 'Alt. splicing'),
        ('Direct protein sequencing', 'Direct prot. seq.'),
        ('Cell membrane', 'PM'),
        ('Plasma membrane', 'PM'),
        ('Cytoplasmic vesicle', 'Cyto. vesicle'),
        ('Cell projection', 'Cell proj.'),
        ('Reference proteome', 'Ref. proteome'),
        ('Proteomics identification', 'Proteomics id.'),
        ('Transmembrane helix', 'TM helix'),
        ('Transmembrane', 'TM'),
        ('Ubl conjugation pathway', 'Ubl conj. path.'),
        ('Ubl conjugation', 'Ubl conj.'),
        ('Pyridine nucleotide biosynthesis', 'Pyridine nucl. biosyn.'),
        ('Lipid metabolism', 'Lipid metab.'),
        ('Fatty acid metabolism', 'FA metab.'),
        ('Chromosome', 'Chrom.'),
        ('Cytoplasm', 'Cyto.'),
        ('Nucleus', 'Nucleus'),
        ('Mitochondrion', 'Mito'),
        ('Extracellular region', 'Extracellular'),
    ]
    t = s
    for a, b in repl:
        t = t.replace(a, b)
    return t


def get_sig_set(df: pd.DataFrame, group: str) -> set:
    sub = df[(df['group'] == group) & (df['corrected_p_value'] < 0.05) & (df['count'] > 0)]
    return set(sub['keyword'].dropna().tolist())


def prep_unique(df: pd.DataFrame, group: str, other_sig: set, topn: int = 20) -> pd.DataFrame:
    sub = df[df['group'] == group].copy()
    if sub.empty:
        return sub
    sub['neglogq'] = -np.log10(sub['corrected_p_value'].replace(0, np.nextafter(0, 1)))
    sig = sub[(sub['corrected_p_value'] < 0.05) & (sub['count'] > 0)].copy()
    if other_sig:
        sig = sig[~sig['keyword'].isin(other_sig)].copy()
    if sig.empty:
        sig = sub.sort_values(['neglogq', 'fold_enrichment', 'percentage'], ascending=[False, False, False]).head(topn)
    else:
        sig = sig.sort_values(['neglogq', 'fold_enrichment', 'percentage'], ascending=[False, False, False]).head(topn)
    sig['label'] = sig['keyword'].apply(abbreviate_keyword).apply(lambda s: shorten(s, 28))
    return sig


def plot_panel(ax, df: pd.DataFrame, title: str, vmin=None, vmax=None):
    if df.empty:
        ax.text(0.5, 0.5, 'No significant keywords', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return None
    df = df.sort_values('neglogq', ascending=False).reset_index(drop=True)
    order = list(df['label'])[::-1]
    x = df['fold_enrichment']
    y = np.arange(len(df))[::-1]
    sizes = df['percentage'] * 8 + 20
    colors = df['neglogq']
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        cmap=BUBBLE_CMAP,
        edgecolors='white',
        linewidths=0.4,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel('Fold Enrichment', fontsize=12, fontweight='bold')
    # remove y-axis label to keep clean
    ax.set_ylabel('')
    # reference line at 1x
    ax.axvline(1.0, ls='--', c=PALETTE['primary'], lw=1.2, alpha=0.6)
    ax.set_title(title, fontweight='bold')
    style_ax(ax)
    ax.tick_params(axis='y', labelsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(10)
    return sc


def main():
    if not os.path.exists(SRC):
        print(f'Missing source: {SRC}')
        return
    df = pd.read_csv(SRC)
    human_sig = get_sig_set(df, 'Human-Specific Essential')
    immune_sig = get_sig_set(df, 'Immune-Specific Essential')

    human = prep_unique(df, 'Human-Specific Essential', other_sig=immune_sig, topn=20)
    immune = prep_unique(df, 'Immune-Specific Essential', other_sig=human_sig, topn=20)

    # compute shared color scale (shorten excessive range)
    all_neg = []
    for sub in [human, immune]:
        if 'neglogq' in sub.columns:
            all_neg.append(sub['neglogq'].values)
    vmax = 10.0
    if len(all_neg) > 0:
        all_neg = np.concatenate(all_neg)
        vmax = float(np.nanpercentile(all_neg, 95))
        vmax = min(10.0, vmax) if np.isfinite(vmax) else 10.0
    vmin = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.8 * 2), sharex=False)
    scatter_handles = []

    sc1 = plot_panel(axes[0], human, 'Human-Specific Essential: Keywords', vmin=vmin, vmax=vmax)
    axes[0].set_title('Human-Specific Essential: Keywords', fontsize=13, fontweight='bold')
    if sc1 is not None:
        scatter_handles.append(sc1)
    axes[0].text(
        0.0,
        1.02,
        'D',
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight='bold',
        ha='left',
        va='bottom',
    )

    sc2 = plot_panel(axes[1], immune, 'Immune-Specific Essential: Keywords', vmin=vmin, vmax=vmax)
    axes[1].set_title('Immune-Specific Essential: Keywords', fontsize=13, fontweight='bold')
    if sc2 is not None:
        scatter_handles.append(sc2)
    axes[1].text(
        0.0,
        1.02,
        'E',
        transform=axes[1].transAxes,
        fontsize=14,
        fontweight='bold',
        ha='left',
        va='bottom',
    )

    sc = scatter_handles[-1] if scatter_handles else None
    if sc is not None:
        plt.subplots_adjust(left=0.32, right=0.88, hspace=0.32)
        fig.canvas.draw()
        p0 = axes[0].get_position()
        pN = axes[-1].get_position()
        y = pN.y0
        h = p0.y1 - pN.y0
        cax = fig.add_axes([0.90, y, 0.02, h])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('-log10(q)', fontsize=13, fontweight='bold')
        cbar.outline.set_linewidth(1.5)
        try:
            ticks = [round(float(vmin), 1), round(float(vmax), 1)]
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=13, width=1.5)
        except Exception:
            cbar.ax.tick_params(labelsize=13, width=1.5)
    else:
        plt.subplots_adjust(left=0.32, right=0.95, hspace=0.28)

    fig.suptitle('Significantly Enriched Keywords', y=0.995, fontsize=14, fontweight='bold')
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
