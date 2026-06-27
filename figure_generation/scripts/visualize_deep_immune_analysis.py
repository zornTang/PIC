#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.colors import LinearSegmentedColormap, Normalize
import textwrap
from Bio.SeqUtils.ProtParam import ProteinAnalysis

warnings.filterwarnings('ignore')

from _fg_paths import FIGURES_NPJ_DIR, PIC2_ANALYSIS_RESULTS, PIC2_PREDICTIONS_DIR
from style_config import apply_style, PALETTE, CMAPS
apply_style()
# Restore all four spines for this script (dense data plots need full frame)
plt.rcParams['axes.spines.top']   = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.grid']         = False

NPJ_OUT = FIGURES_NPJ_DIR


def solid_to_white_cmap(color_hex, name):
    return LinearSegmentedColormap.from_list(name, ["#F8FAFC", color_hex])


def compact_axis_label(text, width=14, max_lines=2):
    text = str(text).strip()
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return text
    wrapped = wrapped[:max_lines]
    if len(wrapped) == max_lines and len(" ".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))) > len(" ".join(wrapped)):
        wrapped[-1] = wrapped[-1].rstrip(".") + "..."
    return "\n".join(wrapped)

COLORS = {
    # 非必需蛋白用极浅灰退到背景，必需蛋白用饱和色突出
    'Both-High Essential':         PALETTE['common'],
    'Immune-only High Essential':  PALETTE['immune'],
    'Human-only High Essential':   PALETTE['human'],
    'Commonly Non-essential':      '#E2E5EA',
    # Legacy aliases
    'Human-Specific Essential':    PALETTE['human'],
    'Immune-Specific Essential':   PALETTE['immune'],
    'Commonly Essential':          PALETTE['common'],
    'Essential':     PALETTE['human'],
    'Non-essential': '#E2E5EA',
    'High':   PALETTE['immune'],
    'Medium': PALETTE['neutral'],
    'Low':    PALETTE['human'],
    'bg':     PALETTE['bg'],
}

# AA properties
AA_WEIGHTS = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19, 
              'G': 75.07, 'H': 155.16, 'I': 131.18, 'K': 146.19, 'L': 131.18, 
              'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20, 
              'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}

AA_GRAVY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 
            'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 
            'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 
            'W': -0.9, 'Y': -1.3}

def calc_mw(seq):
    seq = str(seq).upper().replace('X', '').replace('U', '')
    if not seq: return 0
    return sum(AA_WEIGHTS.get(aa, 110) for aa in seq) - 18.015 * (len(seq) - 1)

def calc_gravy(seq):
    seq = str(seq).upper().replace('X', '').replace('U', '')
    if not seq: return 0
    return sum(AA_GRAVY.get(aa, 0) for aa in seq) / len(seq)
    
def calc_aa_freq(seq):
    seq = str(seq).upper().replace('X', '').replace('U', '')
    if not seq: return {k:0 for k in AA_WEIGHTS.keys()}
    counts = {k: seq.count(k) for k in AA_WEIGHTS.keys()}
    return {k: v/len(seq) for k,v in counts.items()}

def calc_pi(seq):
    seq = str(seq).upper().replace('X', '').replace('U', '').replace('B', '').replace('Z', '')
    if not seq or len(seq) == 0: return 7.0
    try:
        return ProteinAnalysis(seq).isoelectric_point()
    except:
        return 7.0

def calc_charge(seq, ph=7.4):
    seq = str(seq).upper().replace('X', '').replace('U', '').replace('B', '').replace('Z', '')
    if not seq or len(seq) == 0: return 0.0
    try:
        return ProteinAnalysis(seq).charge_at_pH(ph)
    except:
        return 0.0

def calc_hydrophobic_ratio(seq):
    # Standard hydrophobic AAs: A, I, L, M, F, W, V
    hydrophobic_aas = set('AILMFWV')
    seq = str(seq).upper()
    if not seq: return 0.0
    hydro_count = sum(1 for aa in seq if aa in hydrophobic_aas)
    return (hydro_count / len(seq)) * 100.0

def adjust_pvalues_bh(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        bh_value = ranked[i] * n / rank
        prev = min(prev, bh_value)
        adjusted[i] = min(prev, 1.0)

    adjusted_original_order = np.empty(n, dtype=float)
    adjusted_original_order[order] = adjusted
    return adjusted_original_order

class DeepVisualizer:
    def __init__(self, immune_pred_csv, human_pred_csv, domain_csv, keyword_csv, subloc_csv, out_dir):
        self.immune_pred_csv = immune_pred_csv
        self.human_pred_csv = human_pred_csv
        self.domain_csv = domain_csv
        self.keyword_csv = keyword_csv
        self.subloc_csv = subloc_csv
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.load_data()
        
    def load_data(self):
        print("Loading predictions...")
        df_immune = pd.read_csv(self.immune_pred_csv)
        df_human = pd.read_csv(self.human_pred_csv)
        
        # Merge the two prediction files purely by protein_id
        self.df = pd.merge(
            df_immune[['protein_id', 'sequence', 'PES_score', 'prediction', 'confidence']], 
            df_human[['protein_id', 'PES_score', 'prediction', 'confidence']],
            on='protein_id', 
            suffixes=('_immune', '_human')
        )
        
        self.df['MW_kDa'] = self.df['sequence'].apply(lambda x: calc_mw(x) / 1000)
        self.df['GRAVY'] = self.df['sequence'].apply(calc_gravy)
        self.df['pI'] = self.df['sequence'].apply(calc_pi)
        self.df['net_charge'] = self.df['sequence'].apply(lambda x: calc_charge(x, ph=7.4))
        self.df['hydrophobic_aa_pct'] = self.df['sequence'].apply(calc_hydrophobic_ratio)
        
        # New threshold logic
        threshold = 0.8
        
        # Define condition masks based on PES score threshold logic
        immune_high = self.df['PES_score_immune'] >= threshold
        human_high = self.df['PES_score_human'] >= threshold
        
        # Default subgroup
        self.df['subgroup'] = "Commonly Non-essential"
        
        # Assign correctly
        self.df.loc[immune_high & human_high, 'subgroup'] = "Both-High Essential"
        self.df.loc[human_high & ~immune_high, 'subgroup'] = "Human-only High Essential"
        self.df.loc[immune_high & ~human_high, 'subgroup'] = "Immune-only High Essential"
        
        print("Subgroup distribution after classification:")
        print(self.df['subgroup'].value_counts())
            
        print("Loading enrichment data...")
        self.domains_df = pd.read_csv(self.domain_csv) if os.path.exists(self.domain_csv) else None
        self.keywords_df = pd.read_csv(self.keyword_csv) if os.path.exists(self.keyword_csv) else None
        self.subloc_df = pd.read_csv(self.subloc_csv) if os.path.exists(self.subloc_csv) else None
        
    def plot_figure_a_biophysical(self):
        print("Generating Figure A: Physicochemical Properties (2x2 Boxplots)...")
        # Filter data for the boxplots: We only want the 3 target groups (ignore Commonly Non-essential)
        target_groups = ['Human-only High Essential', 'Immune-only High Essential', 'Both-High Essential']
        df_box = self.df[self.df['subgroup'].isin(target_groups)].copy()
        
        # Mapping subgroup names for shorter X-axis labels to match user reference
        group_rename = {
            'Human-only High Essential': 'Human-Specific\nEssential',
            'Immune-only High Essential': 'Immune-Specific\nEssential',
            'Both-High Essential': 'Commonly\nEssential'
        }
        df_box['plot_group'] = df_box['subgroup'].map(group_rename)
        order = ['Human-Specific\nEssential', 'Immune-Specific\nEssential', 'Commonly\nEssential']
        
        # Colors corresponding to subgroups to match user reference palette vibes but using 0_model_comparison theme
        box_palette = {
            'Human-Specific\nEssential': PALETTE['human'],
            'Immune-Specific\nEssential': PALETTE['immune'],
            'Commonly\nEssential': PALETTE['common']
        }
        
        fig = plt.figure(figsize=(6.5, 5.0))
        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)
        
        from scipy import stats
        feature_specs = [
            ('MW_kDa', 'Molecular Weight (kDa)', 'Molecular Weight (kDa)', '(a)'),
            ('pI', 'Isoelectric Point (pI)', 'Isoelectric Point (pI)', '(b)'),
            ('net_charge', 'Net Charge', 'Net Charge', '(c)'),
            ('hydrophobic_aa_pct', 'Hydrophobic AA (%)', 'Hydrophobic AA (%)', '(d)')
        ]

        human_mask = df_box['plot_group'] == 'Human-Specific\nEssential'
        immune_mask = df_box['plot_group'] == 'Immune-Specific\nEssential'

        raw_pvals = {}
        for y_col, _, _, _ in feature_specs:
            group1 = df_box.loc[human_mask, y_col].dropna()
            group2 = df_box.loc[immune_mask, y_col].dropna()
            if len(group1) > 0 and len(group2) > 0:
                _, raw_pvals[y_col] = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            else:
                raw_pvals[y_col] = np.nan

        valid_cols = [col for col, p_val in raw_pvals.items() if not np.isnan(p_val)]
        adjusted_pvals = dict(raw_pvals)
        if valid_cols:
            corrected = adjust_pvalues_bh([raw_pvals[col] for col in valid_cols])
            for col, q_val in zip(valid_cols, corrected):
                adjusted_pvals[col] = q_val

        # Helper for common styling
        def style_boxplot(
            ax,
            y_col,
            ylabel,
            title,
            panel_label,
            show_title=True,
            show_panel_label=True,
            x_tick_labelsize=13,
            x_tick_pad=10,
        ):
            sns.boxplot(data=df_box, x='plot_group', y=y_col, order=order,
                        palette=box_palette, width=0.46, linewidth=2.0,
                        boxprops={'edgecolor': '#1F2933', 'linewidth': 2.0},
                        whiskerprops={'color': '#1F2933', 'linewidth': 1.8},
                        capprops={'color': '#1F2933', 'linewidth': 1.8},
                        medianprops={'color': 'white', 'linewidth': 2.2},
                        flierprops={'marker': 'o', 'markersize': 5.5, 'markeredgecolor': '#333', 'alpha': 0.7},
                        ax=ax)
            ax.set_title(title if show_title else '', fontsize=16, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
            ax.set_xlabel('')
            ax.grid(axis='y', linestyle='-', alpha=0.22, color=PALETTE['grid'])
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', labelsize=x_tick_labelsize, pad=x_tick_pad, width=2.2, length=6.0)
            ax.tick_params(axis='y', labelsize=13, width=2.2, length=6.0)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.3)
                spine.set_color('black')
                
            # Add subplot label (a), (b), etc.
            if show_panel_label:
                ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes, fontsize=12, fontweight='bold')
                
            # Mann-Whitney U test between Human-Specific and Immune-Specific (deeper analysis)
            group1 = df_box[df_box['plot_group'] == 'Human-Specific\nEssential'][y_col].dropna()
            group2 = df_box[df_box['plot_group'] == 'Immune-Specific\nEssential'][y_col].dropna()
            
            if len(group1) > 0 and len(group2) > 0:
                p_val = raw_pvals[y_col]
                q_val = adjusted_pvals[y_col]
                # Draw bracket
                y_max = df_box[y_col].max()
                y_range = y_max - df_box[y_col].min()
                if y_range == 0: y_range = y_max
                h = y_range * 0.05
                bar_y = y_max + h
                
                # Plot line between x=0 (Human) and x=1 (Immune)
                ax.plot([0, 0, 1, 1], [bar_y, bar_y+h/2, bar_y+h/2, bar_y], lw=2.0, c='k')
                
                # Annotate stars
                if q_val < 0.001: sig = '***'
                elif q_val < 0.01: sig = '**'
                elif q_val < 0.05: sig = '*'
                else: sig = 'ns'
                
                ax.text(0.5, bar_y + h*0.8, f"{sig}", ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='black')
                
                # Adjust y-limit
                ax.set_ylim(bottom=df_box[y_col].min() - y_range*0.05, top=bar_y + h*5)
        
        ax1 = fig.add_subplot(gs[0, 0])
        style_boxplot(ax1, *feature_specs[0])

        ax2 = fig.add_subplot(gs[0, 1])
        style_boxplot(ax2, *feature_specs[1])

        ax3 = fig.add_subplot(gs[1, 0])
        style_boxplot(ax3, *feature_specs[2])

        ax4 = fig.add_subplot(gs[1, 1])
        style_boxplot(ax4, *feature_specs[3])

        fig.text(
            0.5, 0.015,
            'Two-sided Mann-Whitney U test; Benjamini-Hochberg FDR correction across 4 panel-wise comparisons',
            ha='center', va='bottom', fontsize=10.5
        )
        
        plt.close()

        NPJ_OUT.mkdir(parents=True, exist_ok=True)
        single_specs = [
            ("fig_13a_biophysical_molecular_weight.png", *feature_specs[0]),
            ("fig_13b_biophysical_isoelectric_point.png", *feature_specs[1]),
            ("fig_13c_biophysical_net_charge.png", *feature_specs[2]),
            ("fig_13d_biophysical_hydrophobic_aa.png", *feature_specs[3]),
        ]
        for filename, y_col, ylabel, title, panel_lab in single_specs:
            single_fig, single_ax = plt.subplots(figsize=(5.6, 4.6))
            style_boxplot(
                single_ax,
                y_col,
                ylabel,
                title,
                panel_lab,
                show_title=False,
                show_panel_label=False,
                x_tick_labelsize=11.5,
                x_tick_pad=8,
            )
            single_fig.tight_layout()
            single_fig.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(single_fig)

    def _plot_bubble(self, df, group_name, title, ax, color_map="mako_r"):
        data = df[(df['group'] == group_name) & (df['significant'] == True)]
        if data.empty:
            ax.set_title(f"No significant data for {group_name}")
            return
            
        data = data.sort_values('p_value').head(15) # Top 15
        data['-log10(p)'] = -np.log10(data['p_value'])
        
        sizes = data['count'] * 20
        scatter = ax.scatter(data['fold_enrichment'], data['-log10(p)'], s=sizes, 
                             c=data['-log10(p)'], cmap=color_map, alpha=0.8, edgecolor='white', linewidth=1)
        
    def plot_figure_b_domains(self):
        if self.domains_df is None: return
        print("Generating Figure B: Domain Enrichment (Multi-panel)...")
        
        # Filter data
        df_sig = self.domains_df[self.domains_df['significant'] == True].copy()
        imm_data = df_sig[df_sig['group'] == 'Immune-Specific Essential']
        hum_data = df_sig[df_sig['group'] == 'Human-Specific Essential']
        com_data = df_sig[df_sig['group'] == 'Commonly Essential']
        
        # Find shared domains (significant in both Immune and Human)
        imm_doms = set(imm_data['domain'])
        hum_doms = set(hum_data['domain'])
        shared_doms = imm_doms.intersection(hum_doms)
        
        # Unique to each
        imm_unique = imm_data[~imm_data['domain'].isin(shared_doms)].sort_values('fold_enrichment', ascending=False).head(15)
        hum_unique = hum_data[~hum_data['domain'].isin(shared_doms)].sort_values('fold_enrichment', ascending=False).head(15)
        
        # Shared (build comparative df)
        shared_data = []
        for dom in shared_doms:
            i_row = imm_data[imm_data['domain'] == dom].iloc[0]
            h_row = hum_data[hum_data['domain'] == dom].iloc[0]
            # Also check Commonly Essential
            c_row = com_data[com_data['domain'] == dom]
            com_fe = c_row['fold_enrichment'].values[0] if not c_row.empty else 0
            avg_fe = (i_row['fold_enrichment'] + h_row['fold_enrichment']) / 2
            shared_data.append({'domain': dom, 'avg_fe': avg_fe, 
                                'imm_fe': i_row['fold_enrichment'], 'hum_fe': h_row['fold_enrichment'],
                                'com_fe': com_fe})
        shared_df = pd.DataFrame(shared_data)
        if not shared_df.empty:
            shared_df = shared_df.sort_values('avg_fe', ascending=False).head(15)
        
        fig = plt.figure(figsize=(8.2, 3.4))
        # (no suptitle per user request)
        gs = fig.add_gridspec(1, 3, wspace=0.55)
        
        # Colors
        imm_color = PALETTE['immune']
        hum_color = PALETTE['human']
        com_color = PALETTE['common']
        
        def format_label(l):
            return compact_axis_label(l, width=14, max_lines=2)
        
        def get_stars(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''
        
        def plot_dot_panel(
            ax,
            data,
            title,
            cmap,
            item_col='domain',
            show_title=True,
            show_panel_label_text=None,
            color_col='p_value',
            colorbar_label='-log10(p)',
            sort_cols=('fold_enrichment',),
            sort_ascending=(True,),
        ):
            if data is None or data.empty:
                ax.set_title(f"No data for {title}")
                return
            data = data.sort_values(list(sort_cols), ascending=list(sort_ascending)).copy()
            data['formatted'] = data[item_col].apply(format_label)
            
            sizes = data['count'].clip(lower=1) * 10
            colors = -np.log10(data[color_col].clip(lower=1e-15))
            
            sc = ax.scatter(data['fold_enrichment'], data['formatted'], 
                            s=sizes, c=colors, cmap=cmap, alpha=0.9, edgecolor='black', linewidth=1.4, zorder=5)
            
            # Lollipop stems: horizontal lines from 0 to the dot
            ax.hlines(y=data['formatted'], xmin=0, xmax=data['fold_enrichment'], color=PALETTE['neutral'], alpha=0.45, linewidth=1.8, zorder=1)
            
            # Reference line at FE = 1 (baseline)
            ax.axvline(x=1, color='black', linestyle=':', linewidth=1.8, alpha=0.6, zorder=0)
            
            ax.set_title(title if show_title else '', fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            ax.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            ax.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            ax.set_xlim(0, data['fold_enrichment'].max() * 1.45)
            cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label(colorbar_label, fontsize=10, fontweight='bold')
            cb.ax.tick_params(labelsize=9, width=1.6, length=4)
            
            # Size legend
            from matplotlib.lines import Line2D
            size_vals = [1, 5, 10]
            size_labels = [f'n={v}' for v in size_vals]
            size_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='grey', 
                                   markersize=np.sqrt(v*25), markeredgecolor='black', label=l) 
                           for v, l in zip(size_vals, size_labels)]
            leg = ax.legend(handles=size_handles, loc='lower right', fontsize=9, title='Count', title_fontsize=9,
                           framealpha=0.88, edgecolor='grey')
            for text in leg.get_texts():
                text.set_fontweight('bold')
            leg.get_title().set_fontweight('bold')
            if show_panel_label_text is not None:
                ax.text(-0.05, 1.08, show_panel_label_text, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        
        # Panel 1: Immune Unique
        ax1 = fig.add_subplot(gs[0, 0])
        plot_dot_panel(ax1, imm_unique, 'Immune-Specific Domains', solid_to_white_cmap(imm_color, "imm_domains"), show_panel_label_text='(a)')
        
        # Panel 2: Human Unique
        ax2 = fig.add_subplot(gs[0, 1])
        plot_dot_panel(ax2, hum_unique, 'Human-Specific Domains', solid_to_white_cmap(hum_color, "hum_domains"), show_panel_label_text='(b)')
        
        # Panel 3: Commonly Essential Domains (only truly enriched ones)
        ax3 = fig.add_subplot(gs[0, 2])
        com_top = com_data[(com_data['fold_enrichment'] > 1) & (com_data['count'] > 0)].sort_values('fold_enrichment', ascending=False).head(15)
        plot_dot_panel(ax3, com_top, 'Commonly Essential Domains', solid_to_white_cmap(com_color, "com_domains"), show_panel_label_text='(c)')
            
        plt.tight_layout()
        plt.close()

        for filename, data, title, cmap in [
            ('fig_14a_domain_enrichment_immune.png', imm_unique, 'Immune-Specific Domains', solid_to_white_cmap(imm_color, "imm_domains_single")),
            ('fig_14b_domain_enrichment_human.png', hum_unique, 'Human-Specific Domains', solid_to_white_cmap(hum_color, "hum_domains_single")),
            ('fig_14c_domain_enrichment_common.png', com_top, 'Commonly Essential Domains', solid_to_white_cmap(com_color, "com_domains_single")),
        ]:
            sf, sa = plt.subplots(figsize=(3.9, 5.4))
            plot_dot_panel(sa, data, title, cmap, show_title=False)
            for t in list(sa.texts):
                if t.get_text() in {'(a)', '(b)', '(c)'}:
                    t.remove()
            sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

        # Reviewer-facing version: use an explicit, mechanical selection rule.
        # This avoids post-hoc selection by showing only enriched terms that pass FDR.
        df_enriched = self.domains_df[
            (self.domains_df['corrected_p_value'] < 0.05)
            & (self.domains_df['fold_enrichment'] > 1)
            & (self.domains_df['count'] > 0)
        ].copy()
        imm_enriched = df_enriched[df_enriched['group'] == 'Immune-Specific Essential']
        hum_enriched = df_enriched[df_enriched['group'] == 'Human-Specific Essential']
        com_enriched = df_enriched[df_enriched['group'] == 'Commonly Essential']

        enriched_shared_doms = set(imm_enriched['domain']).intersection(set(hum_enriched['domain']))
        imm_fdr = (
            imm_enriched[~imm_enriched['domain'].isin(enriched_shared_doms)]
            .sort_values(['corrected_p_value', 'fold_enrichment'], ascending=[True, False])
            .head(15)
        )
        hum_fdr = (
            hum_enriched[~hum_enriched['domain'].isin(enriched_shared_doms)]
            .sort_values(['corrected_p_value', 'fold_enrichment'], ascending=[True, False])
            .head(15)
        )
        com_fdr = (
            com_enriched
            .sort_values(['corrected_p_value', 'fold_enrichment'], ascending=[True, False])
            .head(15)
        )

        for filename, data, title, cmap in [
            ('fig_14a_domain_enrichment_immune_fdr_ranked.png', imm_fdr, 'Immune-Specific Domains', solid_to_white_cmap(imm_color, "imm_domains_fdr_single")),
            ('fig_14b_domain_enrichment_human_fdr_ranked.png', hum_fdr, 'Human-Specific Domains', solid_to_white_cmap(hum_color, "hum_domains_fdr_single")),
            ('fig_14c_domain_enrichment_common_fdr_ranked.png', com_fdr, 'Commonly Essential Domains', solid_to_white_cmap(com_color, "com_domains_fdr_single")),
        ]:
            sf, sa = plt.subplots(figsize=(3.9, 5.4))
            plot_dot_panel(
                sa,
                data,
                title,
                cmap,
                show_title=False,
                color_col='corrected_p_value',
                colorbar_label='-log10(FDR)',
                sort_cols=('corrected_p_value', 'fold_enrichment'),
                sort_ascending=(False, True),
            )
            sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

    def plot_figure_c_keywords(self):
        if self.keywords_df is None: return
        print("Generating Figure C: Keyword Enrichment (Multi-panel)...")
        
        # Filter data
        df_sig = self.keywords_df[self.keywords_df['significant'] == True].copy()
        
        # Exclude biologically irrelevant or misleading keywords
        exclude_keywords = [
            'Neurodegeneration', 'Intellectual disability', 'Deafness', 
            'Meiosis', 'Sodium transport'
        ]
        df_sig = df_sig[~df_sig['keyword'].isin(exclude_keywords)]
        
        imm_data = df_sig[df_sig['group'] == 'Immune-Specific Essential']
        hum_data = df_sig[df_sig['group'] == 'Human-Specific Essential']
        
        # Find shared keywords (significant in both)
        imm_keys = set(imm_data['keyword'])
        hum_keys = set(hum_data['keyword'])
        shared_keys = imm_keys.intersection(hum_keys)
        
        # Unique to each
        imm_unique = imm_data[~imm_data['keyword'].isin(shared_keys)].sort_values(['fold_enrichment'], ascending=False).head(15)
        hum_unique = hum_data[~hum_data['keyword'].isin(shared_keys)].sort_values(['fold_enrichment'], ascending=False).head(15)
        
        # Shared (sort by max fold enrichment in either, or an average)
        shared_data = []
        for kw in shared_keys:
            i_row = imm_data[imm_data['keyword'] == kw].iloc[0]
            h_row = hum_data[hum_data['keyword'] == kw].iloc[0]
            avg_fe = (i_row['fold_enrichment'] + h_row['fold_enrichment']) / 2
            shared_data.append({'keyword': kw, 'avg_fe': avg_fe, 'imm_fe': i_row['fold_enrichment'], 'hum_fe': h_row['fold_enrichment']})
        
        shared_df = pd.DataFrame(shared_data)
        if not shared_df.empty:
            shared_df = shared_df.sort_values('avg_fe', ascending=False).head(15)
        
        fig = plt.figure(figsize=(8.2, 3.4))
        # (no suptitle per user request)
        gs = fig.add_gridspec(1, 3, wspace=0.55)
        
        # Colors matching the theme
        imm_color = PALETTE['immune']
        hum_color = PALETTE['human']
        shared_color = PALETTE['common']
        
        # Helper to format long labels (keep short to save space)
        def format_label(l):
            return compact_axis_label(l, width=14, max_lines=2)
        
        def get_stars(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''
            
        def plot_dot_panel(
            ax,
            data,
            title,
            cmap,
            show_title=True,
            show_panel_label_text=None,
            color_col='p_value',
            colorbar_label='-log10(p)',
            sort_cols=('fold_enrichment',),
            sort_ascending=(True,),
            size_scale=10,
            size_legend_vals=(1, 5, 10),
        ):
            if data is None or data.empty:
                ax.set_title(f"No data for {title}")
                return
            data = data.sort_values(list(sort_cols), ascending=list(sort_ascending)).copy()
            data['formatted'] = data['keyword'].apply(format_label)
            
            # Scatter plot: Size = count, Color = significance metric.
            sizes = data['count'].clip(lower=1) * size_scale
            colors = -np.log10(data[color_col].clip(lower=1e-15))
            
            sc = ax.scatter(data['fold_enrichment'], data['formatted'], 
                            s=sizes, c=colors, cmap=cmap, alpha=0.9, edgecolor='black', linewidth=1.4, zorder=5)
            
            # Lollipop stems
            ax.hlines(y=data['formatted'], xmin=0, xmax=data['fold_enrichment'], color=PALETTE['neutral'], alpha=0.45, linewidth=1.8, zorder=1)
            
            # Reference line at FE = 1
            ax.axvline(x=1, color='black', linestyle=':', linewidth=1.8, alpha=0.6, zorder=0)
            
            ax.set_title(title if show_title else '', fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            ax.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            ax.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            ax.set_xlim(0, data['fold_enrichment'].max() * 1.45)
            
            cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label(colorbar_label, fontsize=10, fontweight='bold')
            cb.ax.tick_params(labelsize=9, width=1.6, length=4)
            
            # Size legend
            from matplotlib.lines import Line2D
            size_vals = list(size_legend_vals)
            size_labels = [f'n={v}' for v in size_vals]
            size_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
                                   markersize=np.sqrt(v * size_scale), markeredgecolor='black', label=l)
                           for v, l in zip(size_vals, size_labels)]
            leg = ax.legend(
                handles=size_handles,
                loc='lower right',
                fontsize=9,
                title='Count',
                title_fontsize=9,
                framealpha=0.8,
                edgecolor='grey',
                borderpad=0.9,
                labelspacing=1.15,
                handleheight=1.8,
            )
            for text in leg.get_texts():
                text.set_fontweight('bold')
            leg.get_title().set_fontweight('bold')
            if show_panel_label_text is not None:
                ax.text(-0.05, 1.08, show_panel_label_text, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

        # Panel 1: Immune Unique (Red Dot Plot)
        ax1 = fig.add_subplot(gs[0, 0])
        plot_dot_panel(ax1, imm_unique, 'Immune-Specific Target Features', solid_to_white_cmap(imm_color, "imm_kw"), show_panel_label_text='(a)')
        
        # Panel 2: Human Unique (Navy Blue Dot Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        plot_dot_panel(
            ax2,
            hum_unique,
            'Human-Specific Core Features',
            solid_to_white_cmap(hum_color, "hum_kw"),
            show_panel_label_text='(b)',
            size_scale=0.8,
            size_legend_vals=(100, 300, 600),
        )
        
        # Panel 3: Shared (Dumbbell Plot)
        ax3 = fig.add_subplot(gs[0, 2])
        if not shared_df.empty:
            shared_df = shared_df.sort_values('avg_fe', ascending=True)
            shared_df['formatted'] = shared_df['keyword'].apply(format_label)
            
            # Draw line bridging the points
            ax3.hlines(y=shared_df['formatted'], xmin=shared_df[['imm_fe','hum_fe']].min(axis=1), 
                       xmax=shared_df[['imm_fe','hum_fe']].max(axis=1), color=PALETTE['neutral'], alpha=0.6, linewidth=1.6, zorder=2)
            
            # Full faint guide lines
            ax3.hlines(y=shared_df['formatted'], xmin=0, xmax=shared_df[['imm_fe','hum_fe']].max(axis=1).max()*1.1, color=PALETTE['neutral'], alpha=0.18, linewidth=1.2, zorder=1)
            
            # Reference line at FE = 1
            ax3.axvline(x=1, color='black', linestyle=':', linewidth=1.8, alpha=0.6, zorder=0)
            
            ax3.scatter(shared_df['hum_fe'], shared_df['formatted'], color=hum_color, s=180, label='Human Essential', zorder=5, edgecolor='white', linewidth=1.2)
            ax3.scatter(shared_df['imm_fe'], shared_df['formatted'], color=imm_color, s=180, label='Immune Essential', zorder=6, edgecolor='white', linewidth=1.2)
            
            for _, row in shared_df.iterrows():
                delta = row['imm_fe'] - row['hum_fe']
                max_fe = max(row['imm_fe'], row['hum_fe'])
                arrow = '→Imm' if delta > 0 else '→Hum'
                color = '#E53E3E' if delta > 0 else '#1A365D'
                ax3.annotate(f"Δ{abs(delta):.1f} {arrow}", (max_fe, row['formatted']),
                            xytext=(8, 0), textcoords='offset points', fontsize=8.5,
                            fontweight='bold', va='center', color=color, zorder=10)
            
            ax3.set_title('Shared Essential Features', fontsize=12, fontweight='bold', pad=15)
            ax3.text(-0.05, 1.08, '(c)', transform=ax3.transAxes, fontsize=11, fontweight='bold', va='top')
            ax3.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
            ax3.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            ax3.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in ax3.get_xticklabels() + ax3.get_yticklabels():
                label.set_fontweight('bold')
            ax3.legend(loc='lower right', fontsize=9, framealpha=0.88)
            
            # Spines - all visible as outer frame
            for spine in ax3.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            ax3.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            ax3.set_xlim(0, shared_df[['imm_fe','hum_fe']].max().max() * 1.4)
            
        plt.tight_layout()
        plt.close()

        for filename, data, title, cmap, size_scale, size_legend_vals in [
            ('fig_15a_keyword_enrichment_immune.png', imm_unique, 'Immune-Specific Target Features', solid_to_white_cmap(imm_color, "imm_kw_single"), 10, (1, 5, 10)),
            ('fig_15b_keyword_enrichment_human.png', hum_unique, 'Human-Specific Core Features', solid_to_white_cmap(hum_color, "hum_kw_single"), 0.8, (100, 300, 600)),
        ]:
            sf, sa = plt.subplots(figsize=(3.9, 5.4))
            plot_dot_panel(
                sa,
                data,
                title,
                cmap,
                show_title=False,
                size_scale=size_scale,
                size_legend_vals=size_legend_vals,
            )
            for t in list(sa.texts):
                if t.get_text() in {'(a)', '(b)'}:
                    t.remove()
            sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

        if not shared_df.empty:
            sf, sa = plt.subplots(figsize=(4.1, 5.6))
            shared_local = shared_df.sort_values('avg_fe', ascending=True).copy()
            shared_local['formatted'] = shared_local['keyword'].apply(format_label)
            sa.hlines(y=shared_local['formatted'], xmin=shared_local[['imm_fe','hum_fe']].min(axis=1),
                      xmax=shared_local[['imm_fe','hum_fe']].max(axis=1), color=PALETTE['neutral'], alpha=0.6, linewidth=1.6, zorder=2)
            sa.hlines(y=shared_local['formatted'], xmin=0, xmax=shared_local[['imm_fe','hum_fe']].max(axis=1).max()*1.1, color=PALETTE['neutral'], alpha=0.18, linewidth=1.2, zorder=1)
            sa.axvline(x=1, color='black', linestyle=':', linewidth=1.8, alpha=0.6, zorder=0)
            sa.scatter(shared_local['hum_fe'], shared_local['formatted'], color=hum_color, s=180, edgecolor='white', linewidth=1.2, label='Human Essential', zorder=5)
            sa.scatter(shared_local['imm_fe'], shared_local['formatted'], color=imm_color, s=180, edgecolor='white', linewidth=1.2, label='Immune Essential', zorder=6)
            for _, row in shared_local.iterrows():
                delta = row['imm_fe'] - row['hum_fe']
                max_fe = max(row['imm_fe'], row['hum_fe'])
                arrow = '→Imm' if delta > 0 else '→Hum'
                color = imm_color if delta > 0 else hum_color
                sa.annotate(f"Δ{abs(delta):.1f} {arrow}", (max_fe, row['formatted']),
                            xytext=(8, 0), textcoords='offset points', fontsize=8.5, fontweight='bold', va='center', color=color, zorder=10)
            sa.set_title('', fontsize=12, fontweight='bold')
            sa.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
            sa.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            sa.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in sa.get_xticklabels() + sa.get_yticklabels():
                label.set_fontweight('bold')
            sa.legend(loc='lower right', fontsize=9, framealpha=0.88)
            for spine in sa.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            sa.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            sa.set_xlim(0, shared_local[['imm_fe','hum_fe']].max().max() * 1.4)
            sf.savefig(NPJ_OUT / 'fig_15c_keyword_enrichment_shared.png', dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

        # Reviewer-facing version: all keywords use the same objective enrichment rule.
        # No biological keyword exclusions are applied in this output.
        df_enriched = self.keywords_df[
            (self.keywords_df['corrected_p_value'] < 0.05)
            & (self.keywords_df['fold_enrichment'] > 1)
            & (self.keywords_df['count'] > 0)
        ].copy()
        imm_enriched = df_enriched[df_enriched['group'] == 'Immune-Specific Essential']
        hum_enriched = df_enriched[df_enriched['group'] == 'Human-Specific Essential']

        enriched_shared_keys = set(imm_enriched['keyword']).intersection(set(hum_enriched['keyword']))
        imm_fdr = (
            imm_enriched[~imm_enriched['keyword'].isin(enriched_shared_keys)]
            .sort_values(['corrected_p_value', 'fold_enrichment'], ascending=[True, False])
            .head(15)
        )
        hum_fdr = (
            hum_enriched[~hum_enriched['keyword'].isin(enriched_shared_keys)]
            .sort_values(['corrected_p_value', 'fold_enrichment'], ascending=[True, False])
            .head(15)
        )

        for filename, data, title, cmap, size_scale, size_legend_vals in [
            ('fig_15a_keyword_enrichment_immune_fdr_ranked.png', imm_fdr, 'Immune-Specific Target Features', solid_to_white_cmap(imm_color, "imm_kw_fdr_single"), 10, (1, 5, 10)),
            ('fig_15b_keyword_enrichment_human_fdr_ranked.png', hum_fdr, 'Human-Specific Core Features', solid_to_white_cmap(hum_color, "hum_kw_fdr_single"), 0.8, (100, 300, 600)),
        ]:
            sf, sa = plt.subplots(figsize=(3.9, 5.4))
            plot_dot_panel(
                sa,
                data,
                title,
                cmap,
                show_title=False,
                color_col='corrected_p_value',
                colorbar_label='-log10(FDR)',
                sort_cols=('corrected_p_value', 'fold_enrichment'),
                sort_ascending=(False, True),
                size_scale=size_scale,
                size_legend_vals=size_legend_vals,
            )
            sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

        shared_fdr_rows = []
        for kw in enriched_shared_keys:
            i_row = imm_enriched[imm_enriched['keyword'] == kw].iloc[0]
            h_row = hum_enriched[hum_enriched['keyword'] == kw].iloc[0]
            shared_fdr_rows.append({
                'keyword': kw,
                'imm_fe': i_row['fold_enrichment'],
                'hum_fe': h_row['fold_enrichment'],
                'imm_count': i_row['count'],
                'hum_count': h_row['count'],
                'max_fdr': max(i_row['corrected_p_value'], h_row['corrected_p_value']),
                'avg_fe': (i_row['fold_enrichment'] + h_row['fold_enrichment']) / 2,
            })
        shared_fdr = pd.DataFrame(shared_fdr_rows)
        if not shared_fdr.empty:
            shared_fdr = (
                shared_fdr
                .sort_values(['max_fdr', 'avg_fe'], ascending=[True, False])
                .head(15)
                .sort_values('avg_fe', ascending=True)
                .copy()
            )
            shared_fdr['formatted'] = shared_fdr['keyword'].apply(format_label)

            sf, sa = plt.subplots(figsize=(4.1, 5.6))
            sa.hlines(y=shared_fdr['formatted'], xmin=shared_fdr[['imm_fe','hum_fe']].min(axis=1),
                      xmax=shared_fdr[['imm_fe','hum_fe']].max(axis=1), color=PALETTE['neutral'], alpha=0.6, linewidth=1.6, zorder=2)
            sa.hlines(y=shared_fdr['formatted'], xmin=0, xmax=shared_fdr[['imm_fe','hum_fe']].max(axis=1).max()*1.1,
                      color=PALETTE['neutral'], alpha=0.18, linewidth=1.2, zorder=1)
            sa.axvline(x=1, color='black', linestyle=':', linewidth=1.8, alpha=0.6, zorder=0)
            sa.scatter(shared_fdr['hum_fe'], shared_fdr['formatted'], color=hum_color, s=shared_fdr['hum_count'].clip(lower=1) * 10,
                       edgecolor='white', linewidth=1.2, label='Human Essential', zorder=5)
            sa.scatter(shared_fdr['imm_fe'], shared_fdr['formatted'], color=imm_color, s=shared_fdr['imm_count'].clip(lower=1) * 10,
                       edgecolor='white', linewidth=1.2, label='Immune Essential', zorder=6)
            for _, row in shared_fdr.iterrows():
                delta = row['imm_fe'] - row['hum_fe']
                max_fe = max(row['imm_fe'], row['hum_fe'])
                arrow = '->Imm' if delta > 0 else '->Hum'
                color = imm_color if delta > 0 else hum_color
                sa.annotate(f"Δ{abs(delta):.1f} {arrow}", (max_fe, row['formatted']),
                            xytext=(8, 0), textcoords='offset points', fontsize=8.5,
                            fontweight='bold', va='center', color=color, zorder=10)
            sa.set_title('', fontsize=12, fontweight='bold')
            sa.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
            sa.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            sa.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in sa.get_xticklabels() + sa.get_yticklabels():
                label.set_fontweight('bold')
            sa.legend(loc='lower right', fontsize=9, framealpha=0.88)
            for spine in sa.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            sa.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            sa.set_xlim(0, shared_fdr[['imm_fe','hum_fe']].max().max() * 1.4)
            sf.savefig(NPJ_OUT / 'fig_15c_keyword_enrichment_shared_fdr_ranked.png', dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

            delta_fdr = shared_fdr.copy()
            delta_fdr['delta_fe'] = delta_fdr['imm_fe'] - delta_fdr['hum_fe']
            delta_fdr = delta_fdr.reindex(delta_fdr['delta_fe'].abs().sort_values(ascending=False).index).head(15)
            delta_fdr = delta_fdr.sort_values('delta_fe', ascending=True).copy()
            delta_fdr['formatted'] = delta_fdr['keyword'].apply(format_label)
            bar_colors = [imm_color if v > 0 else hum_color for v in delta_fdr['delta_fe']]

            sf, sa = plt.subplots(figsize=(4.1, 5.6))
            sa.barh(
                delta_fdr['formatted'],
                delta_fdr['delta_fe'],
                color=bar_colors,
                alpha=0.9,
                edgecolor='black',
                linewidth=1.1,
                height=0.66,
                zorder=3,
            )
            sa.axvline(x=0, color='black', linestyle='-', linewidth=1.8, alpha=0.72, zorder=2)
            sa.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])

            x_abs = max(abs(delta_fdr['delta_fe'].min()), abs(delta_fdr['delta_fe'].max()))
            x_pad = max(0.6, x_abs * 0.22)
            sa.set_xlim(-x_abs - x_pad, x_abs + x_pad)
            for _, row in delta_fdr.iterrows():
                ha = 'left' if row['delta_fe'] >= 0 else 'right'
                offset = x_abs * 0.035 if x_abs else 0.08
                x = row['delta_fe'] + (offset if row['delta_fe'] >= 0 else -offset)
                sa.text(
                    x,
                    row['formatted'],
                    f"{row['delta_fe']:+.1f}",
                    va='center',
                    ha=ha,
                    fontsize=8.5,
                    fontweight='bold',
                    color='#222222',
                    zorder=5,
                )

            sa.text(
                0.02,
                1.025,
                'Human higher',
                transform=sa.transAxes,
                ha='left',
                va='bottom',
                fontsize=9.5,
                fontweight='bold',
                color=hum_color,
            )
            sa.text(
                0.98,
                1.025,
                'Immune higher',
                transform=sa.transAxes,
                ha='right',
                va='bottom',
                fontsize=9.5,
                fontweight='bold',
                color=imm_color,
            )
            sa.set_xlabel('Delta Fold Enrichment (Immune - Human)', fontsize=11, fontweight='bold')
            sa.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            sa.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in sa.get_xticklabels() + sa.get_yticklabels():
                label.set_fontweight('bold')
            for spine in sa.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            sf.savefig(NPJ_OUT / 'fig_15c_keyword_enrichment_shared_fdr_delta.png', dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)

    def plot_figure_d_subloc(self):
        if self.subloc_df is None: return
        print("Generating Figure D: Subcellular Localization (Bar Chart)...")
        
        target_groups = ['Immune-Specific Essential', 'Human-Specific Essential', 'Commonly Essential']
        df = self.subloc_df[self.subloc_df['group'].isin(target_groups)].copy()
        if df.empty: return
        
        # Simplify long location names into major categories
        def simplify_location(loc):
            loc_lower = loc.lower()
            if 'nucleus' in loc_lower or 'chromosome' in loc_lower:
                return 'Nucleus'
            elif 'mitochondri' in loc_lower:
                return 'Mitochondrion'
            elif 'golgi' in loc_lower:
                return 'Golgi apparatus'
            elif 'endoplasmic' in loc_lower:
                return 'Endoplasmic reticulum'
            elif 'lysosome' in loc_lower or 'endosome' in loc_lower:
                return 'Lysosome / Endosome'
            elif 'cell membrane' in loc_lower or 'plasma membrane' in loc_lower:
                return 'Cell membrane'
            elif 'membrane' in loc_lower:
                return 'Membrane (other)'
            elif 'secreted' in loc_lower or 'extracellular' in loc_lower:
                return 'Secreted / Extracellular'
            elif 'vesicle' in loc_lower:
                return 'Vesicle'
            elif 'cilium' in loc_lower or 'cytoskeleton' in loc_lower:
                return 'Cytoskeleton / Cilium'
            elif 'cytoplasm' in loc_lower:
                return 'Cytoplasm'
            else:
                return 'Other'
        
        df['simplified'] = df['location'].apply(simplify_location)
        
        # Aggregate by simplified location per group
        agg = df.groupby(['group', 'simplified']).agg(
            count=('count', 'sum'),
            total=('total', 'max')
        ).reset_index()
        agg['percentage'] = agg['count'] / agg['total'] * 100
        
        # Per-group color schemes (consistent with Figures B & C)
        group_cmaps = {
            'Immune-Specific Essential': solid_to_white_cmap(PALETTE['immune'], 'd_imm'),
            'Human-Specific Essential': solid_to_white_cmap(PALETTE['human'], 'd_hum'),
            'Commonly Essential': solid_to_white_cmap(PALETTE['common'], 'd_com')
        }

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
        fig.subplots_adjust(wspace=0.35)
        # (no suptitle per user request)
        
        panel_config = [
            ('Immune-Specific Essential', 'Immune-Specific Essential', '(a)'),
            ('Human-Specific Essential', 'Human-Specific Essential', '(b)'),
            ('Commonly Essential', 'Commonly Essential', '(c)')
        ]
        
        for ax, (group, title, panel_label) in zip(axes, panel_config):
            gdata = agg[agg['group'] == group].copy()
            gdata = gdata[gdata['count'] > 0].sort_values('count', ascending=True)
            
            if gdata.empty:
                ax.set_title(f"No data for {title}")
                continue
            
            # Generate gradient colors based on within-group proportion (darker = higher proportion).
            cmap = group_cmaps[group]
            norm = Normalize(vmin=gdata['percentage'].min() * 0.3, vmax=gdata['percentage'].max() * 1.1)
            bar_colors = [cmap(norm(v)) for v in gdata['percentage'].values]
            max_pct = gdata['percentage'].max()
            
            bars = ax.barh(gdata['simplified'], gdata['percentage'], color=bar_colors, 
                          edgecolor='white', linewidth=1.2, height=0.7, zorder=3)
            
            # Count/proportion annotations
            for bar, count_val, total_val, pct_val in zip(bars, gdata['count'].values, gdata['total'].values, gdata['percentage'].values):
                ax.text(bar.get_width() + max_pct * 0.025, bar.get_y() + bar.get_height()/2, 
                        f'{pct_val:.1f}% ({int(count_val)}/{int(total_val)})', fontsize=7.8,
                        fontweight='bold', va='center', color='#333333')
            
            ax.set_xlabel('Proportion of Annotated Proteins (%)', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.text(-0.05, 1.08, panel_label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
            
            ax.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            ax.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            
            ax.set_xlim(-0.2, max_pct * 1.5)
            ax.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            
            # Total count in corner
            total = gdata['count'].sum()
            ax.text(0.95, 0.05, f'Total: {total}', transform=ax.transAxes, fontsize=8, 
                    fontweight='bold', ha='right', va='bottom', color='grey',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8))
        
        plt.tight_layout()
        plt.close()

        for filename, group, title in [
            ('fig_16a_subcellular_localization_immune.png', 'Immune-Specific Essential', 'Immune-Specific Essential'),
            ('fig_16b_subcellular_localization_human.png', 'Human-Specific Essential', 'Human-Specific Essential'),
            ('fig_16c_subcellular_localization_common.png', 'Commonly Essential', 'Commonly Essential'),
        ]:
            gdata = agg[agg['group'] == group].copy()
            gdata = gdata[gdata['count'] > 0].sort_values('count', ascending=True)
            if gdata.empty:
                continue
            sf, sa = plt.subplots(figsize=(4.0, 5.4))
            cmap = group_cmaps[group]
            norm = Normalize(vmin=gdata['percentage'].min() * 0.3, vmax=gdata['percentage'].max() * 1.1)
            bar_colors = [cmap(norm(v)) for v in gdata['percentage'].values]
            max_pct = gdata['percentage'].max()
            bars = sa.barh(gdata['simplified'], gdata['percentage'], color=bar_colors, edgecolor='white', linewidth=1.2, height=0.7, zorder=3)
            for bar, count_val, total_val, pct_val in zip(bars, gdata['count'].values, gdata['total'].values, gdata['percentage'].values):
                sa.text(bar.get_width() + max_pct * 0.025, bar.get_y() + bar.get_height()/2,
                        f'{pct_val:.1f}% ({int(count_val)}/{int(total_val)})', fontsize=7.8,
                        fontweight='bold', va='center', color='#333333')
            sa.set_xlabel('Proportion of Annotated Proteins (%)', fontsize=10, fontweight='bold')
            sa.set_title('', fontsize=12, fontweight='bold')
            sa.tick_params(axis='x', labelsize=10, width=2.0, length=5)
            sa.tick_params(axis='y', labelsize=9, width=2.0, length=5)
            for label in sa.get_xticklabels() + sa.get_yticklabels():
                label.set_fontweight('bold')
            for spine in sa.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_color('black')
            sa.set_xlim(-0.2, max_pct * 1.5)
            sa.grid(axis='x', linestyle='--', alpha=0.22, color=PALETTE['grid'])
            sf.savefig(NPJ_OUT / filename, dpi=600, bbox_inches='tight', facecolor='none', transparent=True)
            plt.close(sf)
    def run_all(self):
        self.plot_figure_a_biophysical()
        self.plot_figure_b_domains()
        self.plot_figure_c_keywords()
        self.plot_figure_d_subloc()
        print(f"Selected PPT visualizations saved to {NPJ_OUT}/")

if __name__ == "__main__":
    import sys
    base_dir = str(PIC2_PREDICTIONS_DIR)
    analysis_dir = str(PIC2_ANALYSIS_RESULTS)
    
    # Needs two prediction sets based on the user request
    immune_pred_file = os.path.join(base_dir, "neutrophil_immune_ensemble_predictions.csv")
    human_pred_file = os.path.join(base_dir, "neutrophil_proteins_human_predictions.csv")
    
    domain_file = os.path.join(analysis_dir, "domain_enrichment_results.csv")
    keyword_file = os.path.join(analysis_dir, "keyword_enrichment_results.csv")
    subloc_file = os.path.join(analysis_dir, "subcellular_localization_distribution.csv")
    
    out_dir = str(FIGURES_NPJ_DIR)
    
    if not (os.path.exists(immune_pred_file) and os.path.exists(human_pred_file)):
        print(f"Error: Prediction files not found. Please check paths.")
        sys.exit(1)
        
    viz = DeepVisualizer(immune_pred_file, human_pred_file, domain_file, keyword_file, subloc_file, out_dir)
    viz.run_all()
