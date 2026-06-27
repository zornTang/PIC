"""
Cell Line Essentiality Analysis
================================
Analyse binary gene essentiality labels (Project Score) for four target genes
(ATP6V1B2, ATP6V1A, PLBD1, H2BC11) across tissue types and compare immune
(Haematopoietic and Lymphoid) vs. non-immune cell lines.

Data
----
* cell_data.pkl  : pandas DataFrame (17185 × 328)
  columns: 'index', 'ID', 'length', 'sequence', 'Gene', + 324 cell-line names
  cell-line values: 0 (non-essential) or 1 (essential)
* cell_line_meta_info.csv : columns: cell_line, tissue, cancer_type, cancer_type_detail
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import fisher_exact

warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────────
NATURE_COLORS = {
    'primary_red':    '#E64B35',
    'primary_blue':   '#4DBBD5',
    'primary_green':  '#00A087',
    'primary_orange': '#F39B7F',
    'primary_purple': '#8491B4',
    'light_blue':     '#91D1C2',
    'light_red':      '#F2B5A7',
    'dark_blue':      '#3C5488',
    'dark_red':       '#DC0000',
}

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR  = os.path.join(SCRIPT_DIR, '..', 'results')

CELL_DATA_PATH = '/home/chein/Documents/code/masterproject/PIC/data/cell_data.pkl'
META_PATH      = '/home/chein/Documents/code/masterproject/PIC/data/cell_line_meta_info.csv'

# H2BC11 was renamed from HIST1H2BC in older Project Score data
TARGET_GENES  = ['ATP6V1B2', 'ATP6V1A', 'PLBD1', 'HIST1H2BC']
GENE_DISPLAY  = {'HIST1H2BC': 'H2BC11'}   # display name for plots
IMMUNE_TISSUE = 'Haematopoietic and Lymphoid'
# Predefined immune cell lines (fallback if not in meta)
IMMUNE_CELL_LINES_FALLBACK = [
    'ARH-77', 'IM-9', 'KMS-11', 'L-363', 'LP-1',
    'OCI-AML2', 'OCI-AML3', 'OCI-LY-19', 'OPM-2',
    'ROS-50', 'RPMI-8226', 'SU-DHL-10', 'SU-DHL-5', 'SU-DHL-8',
]

# non-expression metadata columns
META_CELL_COLS = {'index', 'ID', 'length', 'sequence', 'Gene'}


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    """Load cell_data and meta, returning (cell_data, meta, cell_cols)."""
    print(f'[INFO] Loading cell data from {CELL_DATA_PATH} …')
    with open(CELL_DATA_PATH, 'rb') as fh:
        cell_data = pickle.load(fh)
    print(f'[INFO] cell_data shape: {cell_data.shape}')
    print(f'[INFO] Columns (first 10): {list(cell_data.columns[:10])}')

    # detect cell-line columns
    cell_cols = [c for c in cell_data.columns if c not in META_CELL_COLS]
    print(f'[INFO] Cell-line columns detected: {len(cell_cols)}')

    print(f'[INFO] Loading meta from {META_PATH} …')
    meta = pd.read_csv(META_PATH)
    print(f'[INFO] meta shape: {meta.shape}')
    print(f'[INFO] meta columns: {list(meta.columns)}')

    # detect cell_line name column in meta
    cl_col = None
    for candidate in ['cell_line', 'Cell_line', 'Cell Line', 'cell line',
                       'cell_name', 'CELL_LINE']:
        if candidate in meta.columns:
            cl_col = candidate
            break
    if cl_col is None:
        cl_col = meta.columns[0]
        print(f'[WARN] Could not find cell_line column in meta; using "{cl_col}"')
    else:
        print(f'[INFO] Meta cell-line column: "{cl_col}"')

    # detect tissue column
    tissue_col = None
    for candidate in ['tissue', 'Tissue', 'TISSUE']:
        if candidate in meta.columns:
            tissue_col = candidate
            break
    if tissue_col is None:
        tissue_col = meta.columns[1]
        print(f'[WARN] Could not find tissue column; using "{tissue_col}"')
    else:
        print(f'[INFO] Meta tissue column: "{tissue_col}"')

    return cell_data, meta, cell_cols, cl_col, tissue_col


# ── 2. Build essentiality tables ──────────────────────────────────────────────
def detect_gene_col(cell_data: pd.DataFrame) -> str:
    """Return the column that contains gene names."""
    for candidate in ['Gene', 'gene', 'GENE', 'Gene_name', 'gene_name']:
        if candidate in cell_data.columns:
            return candidate
    raise KeyError('Cannot identify Gene name column in cell_data.')


def build_tissue_essentiality(
        cell_data: pd.DataFrame,
        meta: pd.DataFrame,
        cell_cols: list,
        cl_col: str,
        tissue_col: str,
        target_genes: list) -> pd.DataFrame:
    """
    For each tissue, compute mean essentiality rate of each target gene.

    Returns DataFrame: rows=tissue, cols=target genes.
    """
    gene_col = detect_gene_col(cell_data)

    # filter to target genes
    df_genes = cell_data[cell_data[gene_col].isin(target_genes)].copy()
    df_genes = df_genes.set_index(gene_col)

    # intersect cell-line columns with meta
    meta_cl_set = set(meta[cl_col].values)
    avail_cols  = [c for c in cell_cols if c in meta_cl_set]
    missing     = len(cell_cols) - len(avail_cols)
    if missing:
        print(f'[WARN] {missing} cell-line columns not found in meta; they will be ignored.')
    print(f'[INFO] Cell lines with tissue annotation: {len(avail_cols)}')

    # build cell-line → tissue mapping
    cl_tissue = meta.set_index(cl_col)[tissue_col].to_dict()

    rows = {}
    for tissue in sorted(set(cl_tissue.values())):
        tissue_cols = [c for c in avail_cols if cl_tissue.get(c) == tissue]
        if not tissue_cols:
            continue
        tissue_rates = {}
        for gene in target_genes:
            if gene not in df_genes.index:
                tissue_rates[gene] = np.nan
                continue
            vals = df_genes.loc[gene, tissue_cols].astype(float)
            tissue_rates[gene] = vals.mean()
        rows[tissue] = tissue_rates

    df_tissue = pd.DataFrame(rows).T
    # keep only columns (genes) that exist
    df_tissue = df_tissue[[g for g in target_genes if g in df_tissue.columns]]
    print(f'[INFO] Tissue essentiality table shape: {df_tissue.shape}')
    return df_tissue


def get_immune_nonimmune_splits(
        cell_data: pd.DataFrame,
        meta: pd.DataFrame,
        cell_cols: list,
        cl_col: str,
        tissue_col: str) -> tuple:
    """
    Return (immune_cols, non_immune_cols) based on meta tissue annotation.
    Falls back to the predefined list if no meta match is found.
    """
    if tissue_col in meta.columns and cl_col in meta.columns:
        immune_from_meta = meta.loc[
            meta[tissue_col] == IMMUNE_TISSUE, cl_col].tolist()
        immune_cols     = [c for c in cell_cols if c in immune_from_meta]
        non_immune_cols = [c for c in cell_cols if c not in immune_from_meta]
        if immune_cols:
            print(f'[INFO] Immune cell lines from meta: {len(immune_cols)}')
            return immune_cols, non_immune_cols

    # fallback
    print('[WARN] Using predefined fallback immune cell line list.')
    immune_cols     = [c for c in cell_cols if c in IMMUNE_CELL_LINES_FALLBACK]
    non_immune_cols = [c for c in cell_cols if c not in IMMUNE_CELL_LINES_FALLBACK]
    print(f'[INFO] Immune cell lines (fallback): {len(immune_cols)}')
    return immune_cols, non_immune_cols


# ── 3. Figure 1: Heatmap ─────────────────────────────────────────────────────
def plot_tissue_heatmap(df_tissue: pd.DataFrame, out_path: str) -> None:
    """Heatmap: tissues × target genes, sorted by ATP6V1B2 descending."""
    print('[INFO] Plotting tissue essentiality heatmap …')

    sort_gene = 'ATP6V1B2' if 'ATP6V1B2' in df_tissue.columns else df_tissue.columns[0]
    df_plot   = df_tissue.sort_values(sort_gene, ascending=False)

    n_tissues = len(df_plot)
    n_genes   = len(df_plot.columns)

    fig, ax = plt.subplots(figsize=(10, max(6, n_tissues * 0.45)))

    cmap = LinearSegmentedColormap.from_list(
        'essentiality', ['#FFFFFF', '#FFCDD2', '#E53935', '#7B0000'])
    data = df_plot.values.astype(float)
    im   = ax.imshow(data, aspect='auto', cmap=cmap,
                     vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels([GENE_DISPLAY.get(g, g) for g in df_plot.columns],
                       fontsize=11, fontstyle='italic')
    ax.set_yticks(range(n_tissues))
    ax.set_yticklabels(df_plot.index, fontsize=9)

    # cell text + highlight immune row
    immune_idx = list(df_plot.index).index(IMMUNE_TISSUE) \
        if IMMUNE_TISSUE in df_plot.index else None

    for i in range(n_tissues):
        for j in range(n_genes):
            val = data[i, j]
            if np.isnan(val):
                txt = 'N/A'
                text_color = 'gray'
            else:
                txt        = f'{val * 100:.0f}%'
                text_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=text_color)

    # highlight immune row with a bold rectangle border
    if immune_idx is not None:
        rect = plt.Rectangle(
            (-0.5, immune_idx - 0.5), n_genes, 1,
            linewidth=2.5, edgecolor=NATURE_COLORS['primary_red'],
            facecolor='none', zorder=5)
        ax.add_patch(rect)
        ax.set_yticklabels(
            [f'★ {t}' if t == IMMUNE_TISSUE else t
             for t in df_plot.index],
            fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Essentiality Rate', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.set_title('Target Gene Essentiality Rate Across Tissue Types',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Gene', fontsize=11)
    ax.set_ylabel('Tissue Type', fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved heatmap → {out_path}')


# ── 4. Figure 2: ATP6V1B2 tissue bar plot ────────────────────────────────────
def plot_atp6v1b2_tissue_bar(df_tissue: pd.DataFrame, out_path: str) -> None:
    """Bar plot of ATP6V1B2 essentiality rate per tissue."""
    print('[INFO] Plotting ATP6V1B2 tissue bar chart …')
    gene = 'ATP6V1B2'
    if gene not in df_tissue.columns:
        print(f'[WARN] {gene} not in tissue table; skipping bar plot.')
        return

    series = df_tissue[gene].dropna().sort_values(ascending=False)
    avg_others = series[series.index != IMMUNE_TISSUE].mean()

    colors = [
        NATURE_COLORS['primary_red']
        if t == IMMUNE_TISSUE
        else NATURE_COLORS['primary_blue']
        for t in series.index
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(series)), series.values,
                  color=colors, edgecolor='white', linewidth=0.6)

    # reference line for non-immune mean
    ax.axhline(avg_others, color='gray', linestyle='--', linewidth=1.2,
               label=f'Non-immune mean ({avg_others * 100:.1f}%)')

    # value labels
    for bar, val in zip(bars, series.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{val * 100:.1f}%',
                ha='center', va='bottom', fontsize=7.5, rotation=0)

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Essentiality Rate', fontsize=11)
    ax.set_xlabel('Tissue Type', fontsize=11)
    ax.set_ylim(0, min(1.15, series.values.max() + 0.15))
    ax.set_title(f'{gene} Essentiality Rate Across Tissue Types',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=NATURE_COLORS['primary_red'],
                       label=IMMUNE_TISSUE),
        mpatches.Patch(color=NATURE_COLORS['primary_blue'],
                       label='Other tissues'),
        plt.Line2D([0], [0], color='gray', linestyle='--',
                   label=f'Non-immune mean ({avg_others * 100:.1f}%)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved bar plot → {out_path}')


# ── 5. Figure 3: Immune vs. non-immune grouped bar ───────────────────────────
def run_fisher_test(cell_data: pd.DataFrame,
                    gene_col: str,
                    gene: str,
                    immune_cols: list,
                    non_immune_cols: list) -> tuple:
    """
    Fisher exact test for gene essentiality in immune vs non-immune cell lines.

    Returns (odds_ratio, p_value, immune_rate, non_immune_rate).
    """
    row = cell_data[cell_data[gene_col] == gene]
    if row.empty:
        return np.nan, np.nan, np.nan, np.nan

    row = row.iloc[0]
    imm_vals     = row[immune_cols].astype(float)
    non_imm_vals = row[non_immune_cols].astype(float)

    imm_ess     = int(imm_vals.sum())
    imm_non     = int((imm_vals == 0).sum())
    non_ess     = int(non_imm_vals.sum())
    non_non     = int((non_imm_vals == 0).sum())

    contingency = [[imm_ess, imm_non],
                   [non_ess, non_non]]
    try:
        odds_ratio, p_value = fisher_exact(contingency, alternative='two-sided')
    except Exception:
        odds_ratio, p_value = np.nan, np.nan

    n_imm     = len(imm_vals)
    n_non_imm = len(non_imm_vals)
    imm_rate     = imm_ess / n_imm     if n_imm     > 0 else np.nan
    non_imm_rate = non_ess / n_non_imm if n_non_imm > 0 else np.nan

    return odds_ratio, p_value, imm_rate, non_imm_rate


def p_to_stars(p: float) -> str:
    if np.isnan(p):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def plot_immune_comparison(cell_data: pd.DataFrame,
                           immune_cols: list,
                           non_immune_cols: list,
                           target_genes: list,
                           out_path: str) -> None:
    """Grouped bar plot: immune vs non-immune essentiality for all target genes."""
    print('[INFO] Plotting immune vs. non-immune comparison …')
    gene_col = detect_gene_col(cell_data)

    n_genes = len(target_genes)
    x       = np.arange(n_genes)
    width   = 0.35

    imm_rates     = []
    non_imm_rates = []
    fisher_results = []

    for gene in target_genes:
        or_, pval, ir, nir = run_fisher_test(
            cell_data, gene_col, gene, immune_cols, non_immune_cols)
        imm_rates.append(ir     if not np.isnan(ir)  else 0.0)
        non_imm_rates.append(nir if not np.isnan(nir) else 0.0)
        fisher_results.append((or_, pval))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_imm = ax.bar(x - width / 2, imm_rates, width,
                      color=NATURE_COLORS['primary_red'],
                      label='Immune (Haematopoietic & Lymphoid)',
                      edgecolor='white', linewidth=0.6)
    bars_non = ax.bar(x + width / 2, non_imm_rates, width,
                      color=NATURE_COLORS['primary_blue'],
                      label='Non-immune',
                      edgecolor='white', linewidth=0.6)

    # value labels
    for bar, val in zip(bars_imm, imm_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{val * 100:.1f}%',
                ha='center', va='bottom', fontsize=8.5)
    for bar, val in zip(bars_non, non_imm_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{val * 100:.1f}%',
                ha='center', va='bottom', fontsize=8.5)

    # significance annotations
    y_max = max(max(imm_rates), max(non_imm_rates))
    for i, (or_, pval) in enumerate(fisher_results):
        stars = p_to_stars(pval)
        y_ann = y_max + 0.08 + (0.06 if i % 2 == 1 else 0)
        ax.annotate(
            '', xy=(x[i] + width / 2, y_ann - 0.025),
            xytext=(x[i] - width / 2, y_ann - 0.025),
            arrowprops=dict(arrowstyle='-', color='black', lw=1.2))
        ax.text(x[i], y_ann, stars,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    display_labels = [GENE_DISPLAY.get(g, g) for g in target_genes]
    ax.set_xticklabels(display_labels, fontsize=11, fontstyle='italic')
    ax.set_ylabel('Essentiality Rate', fontsize=11)
    ax.set_xlabel('Gene', fontsize=11)
    ax.set_ylim(0, y_max + 0.22)
    ax.set_title('Target Gene Essentiality: Immune vs. Non-immune Cell Lines',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved comparison plot → {out_path}')


# ── 6. Statistics summary ─────────────────────────────────────────────────────
def print_statistics(cell_data: pd.DataFrame,
                     immune_cols: list,
                     non_immune_cols: list,
                     target_genes: list) -> None:
    gene_col = detect_gene_col(cell_data)
    print('\n' + '=' * 70)
    print('Essentiality Statistics: Immune vs. Non-immune Cell Lines')
    print('=' * 70)
    header = (f'{"Gene":<12} | {"Immune Rate":>12} | '
              f'{"Non-immune Rate":>15} | {"Odds Ratio":>10} | {"P-value":>12}')
    print(header)
    print('-' * 70)
    for gene in target_genes:
        or_, pval, ir, nir = run_fisher_test(
            cell_data, gene_col, gene, immune_cols, non_immune_cols)
        ir_str  = f'{ir  * 100:.1f}%' if not np.isnan(ir)  else 'N/A'
        nir_str = f'{nir * 100:.1f}%' if not np.isnan(nir) else 'N/A'
        or_str  = f'{or_:.3f}'        if not np.isnan(or_)  else 'N/A'
        pv_str  = f'{pval:.4e}'       if not np.isnan(pval) else 'N/A'
        print(f'{gene:<12} | {ir_str:>12} | {nir_str:>15} | '
              f'{or_str:>10} | {pv_str:>12}')
    print('=' * 70 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. Load
    cell_data, meta, cell_cols, cl_col, tissue_col = load_data()

    # 2. Tissue essentiality table
    df_tissue = build_tissue_essentiality(
        cell_data, meta, cell_cols, cl_col, tissue_col, TARGET_GENES)

    # Immune / non-immune split
    immune_cols, non_immune_cols = get_immune_nonimmune_splits(
        cell_data, meta, cell_cols, cl_col, tissue_col)

    # 3. Figure 1: heatmap
    heatmap_path = os.path.join(RESULT_DIR, 'celline_essentiality_heatmap.pdf')
    plot_tissue_heatmap(df_tissue, heatmap_path)

    # 4. Figure 2: ATP6V1B2 tissue bar
    bar_path = os.path.join(RESULT_DIR, 'atp6v1b2_tissue_essentiality.pdf')
    plot_atp6v1b2_tissue_bar(df_tissue, bar_path)

    # 5. Figure 3: immune comparison
    cmp_path = os.path.join(RESULT_DIR, 'target_genes_immune_comparison.pdf')
    plot_immune_comparison(
        cell_data, immune_cols, non_immune_cols, TARGET_GENES, cmp_path)

    # 6. Stats
    print_statistics(cell_data, immune_cols, non_immune_cols, TARGET_GENES)

    print('[DONE] celline_essentiality_analysis.py finished successfully.')


if __name__ == '__main__':
    main()
