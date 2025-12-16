#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

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
    "primary": "#254E7B",   # deep blue
    "secondary": "#F08C2E", # warm amber
    "tertiary": "#6AC1B8",  # soft teal
}
MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "matrix_blue",
    ["#EEF3FB", PALETTE["primary"]]
)

DATA_FOUR = 'analysis/data/neutrophil_four_group_classification.csv'
OUT_DIR = 'result/neutrophil_analysis/visualizations'
os.makedirs(OUT_DIR, exist_ok=True)


def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    # Thicker ticks and larger, bold tick labels
    ax.tick_params(width=2.2, length=6, labelsize=12)
    for lbl in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
        try:
            lbl.set_fontweight('bold')
        except Exception:
            pass


def load_four():
    df = pd.read_csv(DATA_FOUR)
    # sanitize
    for col in ('human_prob', 'immune_prob'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['human_prob', 'immune_prob'])


def make_enhanced_composite(df: pd.DataFrame, outpath: str):
    fig = plt.figure(figsize=(12, 7.8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.15, 1.15],
        hspace=0.85,
        wspace=0.6,
    )

    # Panel A: Scatter by threshold (0.8) classification
    axA = fig.add_subplot(gs[0, 0])
    th = 0.8
    h_ess = df['human_prob'] >= th
    i_ess = df['immune_prob'] >= th
    categories = {
        'Both Essential': (h_ess & i_ess),
        'Human-only Essential': (h_ess & ~i_ess),
        'Immune-only Essential': (~h_ess & i_ess),
        'Both Non-essential': (~h_ess & ~i_ess),
    }
    cat_colors = {
        'Both Essential': PALETTE['primary'],
        'Human-only Essential': PALETTE['secondary'],
        'Immune-only Essential': PALETTE['tertiary'],
        'Both Non-essential': PALETTE['primary'],
    }
    cat_markers = {
        'Both Essential': 'o',
        'Human-only Essential': '^',
        'Immune-only Essential': 's',
        'Both Non-essential': 'o',
    }
    for name, mask in categories.items():
        sub = df[mask]
        color = cat_colors[name]
        marker = cat_markers[name]
        if name == 'Both Non-essential':
            axA.scatter(
                sub['immune_prob'],
                sub['human_prob'],
                s=26,
                facecolors='none',
                edgecolors=color,
                marker=marker,
                linewidths=1.0,
                alpha=0.6,
            )
        else:
            axA.scatter(
                sub['immune_prob'],
                sub['human_prob'],
                s=26,
                c=color,
                edgecolors='white',
                linewidths=0.4,
                marker=marker,
                alpha=0.7,
            )
    axA.plot([0, 1], [0, 1], '--', color='k', lw=1.5, alpha=0.6)
    # Restore threshold markers for composite Panel A
    axA.axhline(th, ls=':', c='gray', lw=1.8)
    axA.axvline(th, ls=':', c='gray', lw=1.8)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.set_xlabel('Immune PES', fontsize=12, fontweight='bold')
    axA.set_ylabel('Human PES', fontsize=12, fontweight='bold')
    r, p = stats.pearsonr(df['human_prob'], df['immune_prob'])
    # r, p annotation
    axA.text(0.03, 0.97, f'r={r:.3f}\nP={p:.1e}', transform=axA.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))
    axA.set_title('A. Global Scatter', loc='left', fontweight='bold')
    style_ax(axA)

    # Panel B: PES distributions (overlaid hist)
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(
        df['human_prob'],
        bins=40,
        alpha=0.55,
        density=True,
        color=PALETTE['primary'],
        label='Human',
    )
    axB.hist(
        df['immune_prob'],
        bins=40,
        alpha=0.55,
        density=True,
        color=PALETTE['secondary'],
        label='Immune',
    )
    axB.axvline(0.5, ls=':', c='gray', lw=1.3)
    axB.set_xlabel('PES', fontsize=12, fontweight='bold')
    axB.set_ylabel('Density', fontsize=12, fontweight='bold')
    axB.set_title('B. PES Distributions', loc='left', fontweight='bold')
    axB.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    style_ax(axB)

    # Panel C: Bland-Altman style with intuitive highlights
    axC = fig.add_subplot(gs[1, 0])
    mean_vals = (df['human_prob'] + df['immune_prob']) / 2
    diff = df['human_prob'] - df['immune_prob']
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    axC.fill_between(
        [0, 1],
        lower,
        upper,
        color=PALETTE['tertiary'],
        alpha=0.12,
        label='±1.96 SD band'
    )
    axC.axhline(0, color='k', lw=1.0, ls='--', alpha=0.6)
    axC.axhline(mean_diff, color=PALETTE['secondary'], lw=1.6, label=f'Mean Δ = {mean_diff:.2f}')
    axC.axhline(upper, color=PALETTE['primary'], lw=1.0, ls=':')
    axC.axhline(lower, color=PALETTE['primary'], lw=1.0, ls=':')

    colors = np.where(diff >= 0, PALETTE['secondary'], PALETTE['tertiary'])
    axC.scatter(
        mean_vals,
        diff,
        c=colors,
        s=28,
        alpha=0.65,
        edgecolors='white',
        linewidths=0.3,
    )

    total = len(diff)
    above = np.sum(diff > upper) / total * 100 if total else 0
    below = np.sum(diff < lower) / total * 100 if total else 0
    within = 100 - above - below
    axC.text(
        0.02,
        0.95,
        f"Within band: {within:.1f}%\nAbove: {above:.1f}%\nBelow: {below:.1f}%",
        transform=axC.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85),
    )

    axC.set_xlim(0, 1)
    axC.set_xlabel('Average PES', fontsize=12, fontweight='bold')
    axC.set_ylabel('Difference (Human - Immune)', fontsize=12, fontweight='bold')
    axC.set_title('C. Bland–Altman Summary', loc='left', fontweight='bold')
    axC.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    style_ax(axC)

    # Panel D: 2x2 confusion (threshold 0.5)
    axD = fig.add_subplot(gs[1, 1])
    h_ess = df['human_prob'] >= th
    i_ess = df['immune_prob'] >= th
    # counts
    tp = int((h_ess & i_ess).sum())
    tn = int((~h_ess & ~i_ess).sum())
    fp = int((~h_ess & i_ess).sum())  # immune essential only
    fn = int((h_ess & ~i_ess).sum())  # human essential only
    mat = np.array([[tp, fn], [fp, tn]])
    im = axD.imshow(mat, cmap=MATRIX_CMAP)
    for (ii, jj), val in np.ndenumerate(mat):
        axD.text(jj, ii, str(val), ha='center', va='center', fontsize=11, fontweight='bold')
    axD.set_xticks([0, 1]); axD.set_yticks([0, 1])
    axD.set_xticklabels(['Immune\nEss', 'Immune\nNon'], fontsize=10)
    axD.set_yticklabels(['Human Ess', 'Human Non'], fontsize=10)
    axD.set_xlabel('Immune classification', fontsize=12, fontweight='bold')
    axD.set_ylabel('Human classification', fontsize=12, fontweight='bold')
    axD.set_title('D. Agreement at 0.8 threshold', loc='left', fontweight='bold')
    style_ax(axD)
    cbar = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(1.5)

    fig.suptitle('Human vs Immune Model Differences', y=0.99, fontsize=12.5, fontweight='bold')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def make_thick_scatter(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    order = ['Both Non-essential', 'Human-only Essential', 'Immune-only Essential', 'Both Essential']
    cat_colors = {
        'Both Essential': PALETTE['primary'],
        'Human-only Essential': PALETTE['secondary'],
        'Immune-only Essential': PALETTE['tertiary'],
        'Both Non-essential': PALETTE['primary'],
    }
    cat_markers = {
        'Both Essential': 'o',
        'Human-only Essential': '^',
        'Immune-only Essential': 's',
        'Both Non-essential': 'o',
    }
    th = 0.8
    h_ess = df['human_prob'] >= th
    i_ess = df['immune_prob'] >= th
    categories = [
        ('Both Essential', (h_ess & i_ess)),
        ('Human-only Essential', (h_ess & ~i_ess)),
        ('Immune-only Essential', (~h_ess & i_ess)),
        ('Both Non-essential', (~h_ess & ~i_ess)),
    ]
    for name, mask in categories:
        sub = df[mask]
        color = cat_colors[name]
        marker = cat_markers[name]
        if name == 'Both Non-essential':
            ax.scatter(
                sub['immune_prob'],
                sub['human_prob'],
                s=28,
                facecolors='none',
                edgecolors=color,
                linewidths=1.0,
                marker=marker,
                alpha=0.6,
            )
        else:
            ax.scatter(
                sub['immune_prob'],
                sub['human_prob'],
                s=28,
                c=color,
                edgecolors='white',
                linewidths=0.4,
                marker=marker,
                alpha=0.7,
            )
    ax.plot([0, 1], [0, 1], '--', color='k', lw=1.8, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Immune PES', fontsize=12, fontweight='bold'); ax.set_ylabel('Human PES', fontsize=12, fontweight='bold')
    # Centered title as requested
    ax.set_title('Human vs Immune PES', fontweight='bold')
    # legend: categories + r/p only (placed outside top-left)
    r, p = stats.pearsonr(df['human_prob'], df['immune_prob'])
    from matplotlib.lines import Line2D
    cat_labels = [name for name, _ in categories]
    legend_handles = []
    for label in cat_labels:
        color = cat_colors[label]
        marker = cat_markers[label]
        if label == 'Both Non-essential':
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markersize=6,
                    linestyle='',
                )
            )
        else:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color='none',
                    markerfacecolor=color,
                    markeredgecolor='white',
                    markeredgewidth=0.4,
                    markersize=6,
                    linestyle='',
                )
            )
    legend_handles.append(Line2D([], [], color='none'))
    legend_labels = cat_labels + [f'Pearson r={r:.3f}, P={p:.1e}']
    ax.legend(legend_handles, legend_labels, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9, markerscale=1.4)
    style_ax(ax)
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def main():
    df = load_four()
    make_enhanced_composite(df, os.path.join(OUT_DIR, 'models_difference_enhanced.png'))
    make_thick_scatter(df, os.path.join(OUT_DIR, 'A_overall_scatter_tj.png'))
    print(f"Saved enhanced figures to {OUT_DIR}")


if __name__ == '__main__':
    main()
