#!/usr/bin/env python3
"""
比较人类层面和免疫细胞层面的必需蛋白预测结果
Enhanced visualization following project standards
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
from collections import Counter
import re
warnings.filterwarnings('ignore')

# 设置专业的Nature风格配色方案
NATURE_COLORS = {
    'primary_red': '#E64B35',
    'primary_blue': '#4DBBD5',
    'primary_green': '#00A087',
    'primary_orange': '#F39B7F',
    'primary_purple': '#8491B4',
    'light_blue': '#91D1C2',
    'light_red': '#F2B5A7',
    'light_green': '#B3E5D1',
    'dark_blue': '#3C5488',
    'dark_red': '#DC0000'
}

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 氨基酸理化性质字典
AMINO_ACID_PROPERTIES = {
    # 分子量 (Da)
    'molecular_weight': {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
        'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
    },
    # Kyte-Doolittle疏水性指数
    'hydrophobicity': {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    },
    # 电荷 (pH 7.0)
    'charge': {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    },
    # 极性
    'polarity': {
        'A': 'nonpolar', 'R': 'polar', 'N': 'polar', 'D': 'polar', 'C': 'polar',
        'Q': 'polar', 'E': 'polar', 'G': 'nonpolar', 'H': 'polar', 'I': 'nonpolar',
        'L': 'nonpolar', 'K': 'polar', 'M': 'nonpolar', 'F': 'nonpolar', 'P': 'nonpolar',
        'S': 'polar', 'T': 'polar', 'W': 'nonpolar', 'Y': 'polar', 'V': 'nonpolar'
    }
}

def calculate_physicochemical_properties(sequence):
    """计算蛋白质序列的理化性质"""
    if pd.isna(sequence) or not sequence:
        return None

    # 清理序列，只保留标准氨基酸
    clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())

    if not clean_sequence:
        return None

    length = len(clean_sequence)
    aa_counts = Counter(clean_sequence)

    # 计算氨基酸组成百分比
    aa_composition = {aa: count/length*100 for aa, count in aa_counts.items()}

    # 计算分子量
    molecular_weight = sum(AMINO_ACID_PROPERTIES['molecular_weight'].get(aa, 0) * count
                          for aa, count in aa_counts.items())

    # 计算平均疏水性
    hydrophobicity = sum(AMINO_ACID_PROPERTIES['hydrophobicity'].get(aa, 0) * count
                        for aa, count in aa_counts.items()) / length

    # 计算净电荷
    net_charge = sum(AMINO_ACID_PROPERTIES['charge'].get(aa, 0) * count
                    for aa, count in aa_counts.items())

    # 计算等电点 (简化计算)
    positive_charge = sum(count for aa, count in aa_counts.items()
                         if AMINO_ACID_PROPERTIES['charge'].get(aa, 0) > 0)
    negative_charge = sum(count for aa, count in aa_counts.items()
                         if AMINO_ACID_PROPERTIES['charge'].get(aa, 0) < 0)

    # 简化的等电点估算
    if positive_charge > negative_charge:
        isoelectric_point = 7.0 + (positive_charge - negative_charge) / length * 5
    elif negative_charge > positive_charge:
        isoelectric_point = 7.0 - (negative_charge - positive_charge) / length * 5
    else:
        isoelectric_point = 7.0

    # 计算极性氨基酸比例
    polar_count = sum(count for aa, count in aa_counts.items()
                     if AMINO_ACID_PROPERTIES['polarity'].get(aa) == 'polar')
    polarity_ratio = polar_count / length * 100

    # 计算特殊氨基酸比例
    aromatic_aa = ['F', 'W', 'Y']
    aromatic_ratio = sum(aa_counts.get(aa, 0) for aa in aromatic_aa) / length * 100

    hydrophobic_aa = ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    hydrophobic_ratio = sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa) / length * 100

    charged_aa = ['R', 'K', 'D', 'E']
    charged_ratio = sum(aa_counts.get(aa, 0) for aa in charged_aa) / length * 100

    return {
        'length': length,
        'molecular_weight': molecular_weight,
        'hydrophobicity': hydrophobicity,
        'net_charge': net_charge,
        'isoelectric_point': isoelectric_point,
        'polarity_ratio': polarity_ratio,
        'aromatic_ratio': aromatic_ratio,
        'hydrophobic_ratio': hydrophobic_ratio,
        'charged_ratio': charged_ratio,
        'aa_composition': aa_composition
    }

def add_physicochemical_analysis(df):
    """为数据框添加理化性质分析"""
    print("Computing physicochemical properties...")

    # 计算理化性质
    physchem_data = []
    for idx, row in df.iterrows():
        props = calculate_physicochemical_properties(row.get('sequence', ''))
        if props:
            physchem_data.append(props)
        else:
            # 如果无法计算，填充NaN
            physchem_data.append({
                'length': np.nan, 'molecular_weight': np.nan, 'hydrophobicity': np.nan,
                'net_charge': np.nan, 'isoelectric_point': np.nan, 'polarity_ratio': np.nan,
                'aromatic_ratio': np.nan, 'hydrophobic_ratio': np.nan, 'charged_ratio': np.nan,
                'aa_composition': {}
            })

    # 将理化性质添加到数据框
    physchem_df = pd.DataFrame(physchem_data)

    # 合并数据
    result_df = pd.concat([df.reset_index(drop=True), physchem_df], axis=1)

    return result_df

# 二硫键分析功能已移至独立模块: disulfide_bond_analysis.py
# from disulfide_bond_analysis import add_disulfide_analysis, analyze_disulfide_bonds, create_disulfide_bond_analysis_visualization

def load_and_clean_data():
    """加载和清理数据"""
    # 读取人类层面预测结果
    human_df = pd.read_csv('../data/neutrophil_proteins_human_predictions.csv')

    # 读取免疫细胞层面预测结果
    immune_df = pd.read_csv('../data/neutrophil_immune_ensemble_predictions.csv')
    
    # 提取蛋白质ID的基因名称部分用于匹配
    def extract_gene_name(protein_id):
        try:
            parts = protein_id.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith('Gene:'):
                    return part.replace('Gene:', '').strip()
            # 如果没有找到Gene:标签，使用原逻辑作为备用
            for part in parts:
                part = part.strip()
                if not part.startswith('ENS') and not part.startswith('OTT') and not part.startswith('GENCODE') and len(part) > 2:
                    return part
            return parts[-1].strip() if parts else protein_id
        except:
            return protein_id
    
    human_df['gene_name'] = human_df['protein_id'].apply(extract_gene_name)
    immune_df['gene_name'] = immune_df['protein_id'].apply(extract_gene_name)

    # 添加理化性质分析
    print("📊 Adding physicochemical analysis to human-level data...")
    human_df = add_physicochemical_analysis(human_df)

    print("📊 Adding physicochemical analysis to immune-level data...")
    immune_df = add_physicochemical_analysis(immune_df)

    # 添加二硫键分析
    # 二硫键分析已移至独立模块
    # print("🔗 Adding disulfide bond analysis to human-level data...")
    # human_df = add_disulfide_analysis(human_df)

    # print("🔗 Adding disulfide bond analysis to immune-level data...")
    # immune_df = add_disulfide_analysis(immune_df)

    return human_df, immune_df

def compare_predictions(human_df, immune_df):
    """对比两种预测结果"""
    # 获取理化性质列和二硫键分析列
    physchem_cols = ['molecular_weight', 'hydrophobicity', 'net_charge', 'isoelectric_point',
                     'polarity_ratio', 'aromatic_ratio', 'hydrophobic_ratio', 'charged_ratio', 'aa_composition']

    # 二硫键分析列已移至独立模块
    # disulfide_cols = ['cys_count', 'potential_disulfide_bonds', 'disulfide_probability', 'cys_percentage',
    #                   'min_distance', 'max_distance', 'avg_distance', 'bond_distribution']

    # 准备合并的列
    human_cols = ['protein_id', 'gene_name', 'PES_score', 'prediction', 'confidence']
    immune_cols = ['protein_id', 'gene_name', 'PES_score', 'prediction', 'confidence']

    # 添加理化性质列（二硫键分析列已移至独立模块）
    all_extra_cols = physchem_cols  # + disulfide_cols
    for col in all_extra_cols:
        if col in human_df.columns:
            human_cols.append(col)
        if col in immune_df.columns:
            immune_cols.append(col)

    # 合并数据集
    merged_df = pd.merge(
        human_df[human_cols],
        immune_df[immune_cols],
        on='protein_id',
        suffixes=('_human', '_immune'),
        how='inner'
    )
    
    print(f"Successfully matched proteins: {len(merged_df)}")

    # 调试：检查合并后的列
    print(f"Merged dataframe columns: {list(merged_df.columns)}")

    # 基本统计
    print("\n=== Basic Statistics ===")
    print(f"Human Level Predictions:")
    print(f"  Total proteins: {len(human_df)}")
    print(f"  Essential proteins: {len(human_df[human_df['prediction'] == 'Essential'])}")
    print(f"  Non-essential proteins: {len(human_df[human_df['prediction'] == 'Non-essential'])}")
    print(f"  Essential protein ratio: {len(human_df[human_df['prediction'] == 'Essential'])/len(human_df)*100:.1f}%")
    
    print(f"\nImmune Cell Level Predictions:")
    print(f"  Total proteins: {len(immune_df)}")
    print(f"  Essential proteins: {len(immune_df[immune_df['prediction'] == 'Essential'])}")
    print(f"  Non-essential proteins: {len(immune_df[immune_df['prediction'] == 'Non-essential'])}")
    print(f"  Essential protein ratio: {len(immune_df[immune_df['prediction'] == 'Essential'])/len(immune_df)*100:.1f}%")
    
    return merged_df

def analyze_agreement(merged_df):
    """分析预测一致性"""
    print("\n=== Prediction Agreement Analysis ===")
    
    # 计算一致性
    agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
    agreement_rate = agreement.sum() / len(merged_df) * 100
    
    print(f"Proteins with consistent predictions: {agreement.sum()}")
    print(f"Prediction agreement rate: {agreement_rate:.1f}%")
    
    # 混淆矩阵
    confusion_matrix = pd.crosstab(
        merged_df['prediction_human'], 
        merged_df['prediction_immune'],
        margins=True
    )
    print(f"\nConfusion Matrix:")
    print(confusion_matrix)
    
    # 分析不一致的情况
    disagreement = merged_df[~agreement]
    print(f"\nProteins with inconsistent predictions: {len(disagreement)}")
    
    if len(disagreement) > 0:
        print("\nDisagreement breakdown:")
        human_essential_immune_non = len(disagreement[
            (disagreement['prediction_human'] == 'Essential') & 
            (disagreement['prediction_immune'] == 'Non-essential')
        ])
        human_non_immune_essential = len(disagreement[
            (disagreement['prediction_human'] == 'Non-essential') & 
            (disagreement['prediction_immune'] == 'Essential')
        ])
        
        print(f"  Human Essential + Immune Non-essential: {human_essential_immune_non}")
        print(f"  Human Non-essential + Immune Essential: {human_non_immune_essential}")
    
    return disagreement

def analyze_score_correlation(merged_df):
    """分析PES分数相关性"""
    print("\n=== PES Score Correlation Analysis ===")
    
    correlation = stats.pearsonr(merged_df['PES_score_human'], merged_df['PES_score_immune'])
    spearman_corr = stats.spearmanr(merged_df['PES_score_human'], merged_df['PES_score_immune'])
    
    print(f"Pearson correlation: {correlation[0]:.3f} (p-value: {correlation[1]:.2e})")
    print(f"Spearman correlation: {spearman_corr[0]:.3f} (p-value: {spearman_corr[1]:.2e})")
    
    # 分数差异分析
    score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']
    print(f"\nPES Score Difference Statistics:")
    print(f"  Mean difference: {score_diff.mean():.3f}")
    print(f"  Standard deviation: {score_diff.std():.3f}")
    print(f"  Maximum positive difference: {score_diff.max():.3f}")
    print(f"  Maximum negative difference: {score_diff.min():.3f}")

def analyze_physicochemical_differences(merged_df):
    """分析不同预测组之间的理化性质差异"""
    print("\n=== Physicochemical Properties Analysis ===")

    # 定义理化性质列（带后缀）
    physchem_cols = ['molecular_weight_human', 'hydrophobicity_human', 'net_charge_human', 'isoelectric_point_human',
                     'polarity_ratio_human', 'aromatic_ratio_human', 'hydrophobic_ratio_human', 'charged_ratio_human',
                     'molecular_weight_immune', 'hydrophobicity_immune', 'net_charge_immune', 'isoelectric_point_immune',
                     'polarity_ratio_immune', 'aromatic_ratio_immune', 'hydrophobic_ratio_immune', 'charged_ratio_immune']

    # 创建预测组合标签
    merged_df['prediction_combo'] = merged_df['prediction_human'] + ' vs ' + merged_df['prediction_immune']

    # 基于高置信度阈值（PES ≥ 0.8）重新定义分组
    high_confidence_threshold = 0.8

    # 定义新的分组
    high_conf_both = merged_df[
        (merged_df['PES_score_human'] >= high_confidence_threshold) &
        (merged_df['PES_score_immune'] >= high_confidence_threshold)
    ]

    high_conf_human_only = merged_df[
        (merged_df['PES_score_human'] >= high_confidence_threshold) &
        (merged_df['PES_score_immune'] < high_confidence_threshold)
    ]

    high_conf_immune_only = merged_df[
        (merged_df['PES_score_human'] < high_confidence_threshold) &
        (merged_df['PES_score_immune'] >= high_confidence_threshold)
    ]

    low_conf_both = merged_df[
        (merged_df['PES_score_human'] < high_confidence_threshold) &
        (merged_df['PES_score_immune'] < high_confidence_threshold)
    ]

    groups = [
        ("High-confidence Both (≥0.8 & ≥0.8)", high_conf_both),
        ("Human-specific High-confidence (≥0.8 & <0.8)", high_conf_human_only),
        ("Immune-specific High-confidence (<0.8 & ≥0.8)", high_conf_immune_only),
        ("Low-confidence Both (<0.8 & <0.8)", low_conf_both)
    ]

    print("\nPhysicochemical Properties by High-Confidence Groups (PES ≥ 0.8):")
    print("=" * 70)

    for group_name, group_data in groups:
        if len(group_data) > 0:
            print(f"\n{group_name} (n={len(group_data)}):")
            # 分别显示人类和免疫层面的理化性质
            human_props = ['molecular_weight_human', 'hydrophobicity_human', 'net_charge_human', 'isoelectric_point_human']
            immune_props = ['molecular_weight_immune', 'hydrophobicity_immune', 'net_charge_immune', 'isoelectric_point_immune']

            print("  Human level properties:")
            for prop in human_props:
                if prop in group_data.columns:
                    mean_val = group_data[prop].mean()
                    std_val = group_data[prop].std()
                    if not pd.isna(mean_val):
                        prop_name = prop.replace('_human', '')
                        print(f"    {prop_name}: {mean_val:.3f} ± {std_val:.3f}")

            print("  Immune level properties:")
            for prop in immune_props:
                if prop in group_data.columns:
                    mean_val = group_data[prop].mean()
                    std_val = group_data[prop].std()
                    if not pd.isna(mean_val):
                        prop_name = prop.replace('_immune', '')
                        print(f"    {prop_name}: {mean_val:.3f} ± {std_val:.3f}")

    # 比较人类特异性vs免疫特异性高置信度蛋白质的理化性质
    print("\n" + "=" * 70)
    print("COMPARISON: Human-specific vs Immune-specific High-Confidence Proteins")
    print("=" * 70)

    human_specific = high_conf_human_only
    immune_specific = high_conf_immune_only

    if len(human_specific) > 0 and len(immune_specific) > 0:
        print(f"\nHuman-specific essential proteins (n={len(human_specific)}):")
        human_props = ['molecular_weight_human', 'hydrophobicity_human', 'net_charge_human', 'isoelectric_point_human',
                       'polarity_ratio_human', 'aromatic_ratio_human', 'hydrophobic_ratio_human', 'charged_ratio_human']

        for prop in human_props:
            if prop in human_specific.columns:
                mean_val = human_specific[prop].mean()
                std_val = human_specific[prop].std()
                if not pd.isna(mean_val):
                    prop_name = prop.replace('_human', '')
                    print(f"  {prop_name}: {mean_val:.3f} ± {std_val:.3f}")

        print(f"\nImmune-specific essential proteins (n={len(immune_specific)}):")
        immune_props = ['molecular_weight_immune', 'hydrophobicity_immune', 'net_charge_immune', 'isoelectric_point_immune',
                        'polarity_ratio_immune', 'aromatic_ratio_immune', 'hydrophobic_ratio_immune', 'charged_ratio_immune']

        for prop in immune_props:
            if prop in immune_specific.columns:
                mean_val = immune_specific[prop].mean()
                std_val = immune_specific[prop].std()
                if not pd.isna(mean_val):
                    prop_name = prop.replace('_immune', '')
                    print(f"  {prop_name}: {mean_val:.3f} ± {std_val:.3f}")

        # 统计显著性检验（比较相同性质的人类和免疫特异性）
        print(f"\nStatistical Comparisons (Human-specific vs Immune-specific):")
        base_props = ['molecular_weight', 'hydrophobicity', 'net_charge', 'isoelectric_point',
                      'polarity_ratio', 'aromatic_ratio', 'hydrophobic_ratio', 'charged_ratio']

        for prop in base_props:
            human_prop = f"{prop}_human"
            immune_prop = f"{prop}_immune"

            if human_prop in human_specific.columns and immune_prop in immune_specific.columns:
                human_vals = human_specific[human_prop].dropna()
                immune_vals = immune_specific[immune_prop].dropna()

                if len(human_vals) > 5 and len(immune_vals) > 5:
                    try:
                        stat, p_value = stats.ttest_ind(human_vals, immune_vals)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        print(f"  {prop}: p={p_value:.2e} {significance}")
                    except:
                        print(f"  {prop}: Unable to compute")

    return human_specific, immune_specific

# analyze_disulfide_bonds 函数已移至 disulfide_bond_analysis.py 模块

def get_protein_function_annotation(protein_id):
    """获取蛋白质功能注释"""
    protein_id = protein_id.upper()

    # 基于已知的功能模式进行注释
    function_map = {
        # 转录调节因子
        'CREB5': '转录调节因子 - cAMP反应元件结合蛋白',
        'SOX6': '转录因子 - SRY相关HMG-box蛋白',
        'TLE3': '转录共抑制因子 - Groucho家族',

        # 表观遗传调节
        'SMCHD1': '表观遗传调节 - 结构维持染色体蛋白',
        'WDFY3': '自噬相关 - WD重复和FYVE结构域蛋白',

        # 组蛋白相关
        'H2BC4': '组蛋白 - 核心组蛋白H2B',
        'H2BC11': '组蛋白 - 核心组蛋白H2B',
        'H2BC18': '组蛋白 - 核心组蛋白H2B',
        'H3-7': '组蛋白 - 核心组蛋白H3',
        'UBN1': '组蛋白结合 - 泛素样PHD和环指结构域蛋白',

        # 细胞周期和增殖
        'HYCC2': '细胞周期调节 - 细胞周期检查点蛋白',

        # 核糖体蛋白
        'RPS4Y2': '核糖体蛋白 - 40S核糖体蛋白S4Y2',

        # 代谢相关
        'NAMPT': '代谢酶 - 烟酰胺磷酸核糖转移酶',
        'SOD2': '抗氧化酶 - 超氧化物歧化酶2',
        'ACSL1': '代谢酶 - 酰基辅酶A合成酶',
        'MME': '膜金属内肽酶 - 信号传导',
        'NPR3': '利钠肽受体 - 信号传导'
    }

    # 提取基因名称
    for gene in function_map.keys():
        if gene in protein_id:
            return function_map[gene]

    # 基于模式匹配的通用分类
    if any(term in protein_id for term in ['RPS', 'RPL']):
        return '核糖体蛋白 - 蛋白质合成'
    elif any(term in protein_id for term in ['H2B', 'H3', 'H4', 'HIST']):
        return '组蛋白 - 染色质结构'
    elif any(term in protein_id for term in ['CREB', 'SOX', 'TLE']):
        return '转录调节 - 基因表达控制'
    elif any(term in protein_id for term in ['ACSL', 'MME']):
        return '代谢相关 - 细胞代谢'
    else:
        return '其他功能'

def identify_interesting_proteins(merged_df, disagreement):
    """识别有趣的蛋白质 - 增强版"""
    print("\n=== Proteins of Special Interest ===")

    # 使用高置信度分组逻辑（PES ≥ 0.8）
    high_confidence_threshold = 0.8

    # 人类特异性高置信度蛋白质
    human_specific = merged_df[
        (merged_df['PES_score_human'] >= high_confidence_threshold) &
        (merged_df['PES_score_immune'] < high_confidence_threshold)
    ].sort_values('PES_score_human', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    if len(human_specific) > 0:
        print(f"\nHuman-specific essential proteins (top 10):")
        print("=" * 80)
        for i, row in human_specific.head(10).iterrows():
            gene_name = row['gene_name_human']
            function = get_protein_function_annotation(row['protein_id'])
            score_diff = row['PES_score_human'] - row['PES_score_immune']
            print(f"  {i+1:2d}. {gene_name}")
            print(f"      功能: {function}")
            print(f"      人类PES: {row['PES_score_human']:.3f} | 免疫PES: {row['PES_score_immune']:.3f} | 差异: +{score_diff:.3f}")
            print()

    # 免疫特异性高置信度蛋白质
    immune_specific = merged_df[
        (merged_df['PES_score_human'] < high_confidence_threshold) &
        (merged_df['PES_score_immune'] >= high_confidence_threshold)
    ].sort_values('PES_score_immune', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    if len(immune_specific) > 0:
        print(f"\nImmune-specific essential proteins (top 10):")
        print("=" * 80)
        for i, row in immune_specific.head(10).iterrows():
            gene_name = row['gene_name_immune']
            function = get_protein_function_annotation(row['protein_id'])
            score_diff = row['PES_score_human'] - row['PES_score_immune']
            print(f"  {i+1:2d}. {gene_name}")
            print(f"      功能: {function}")
            print(f"      人类PES: {row['PES_score_human']:.3f} | 免疫PES: {row['PES_score_immune']:.3f} | 差异: {score_diff:.3f}")
            print()

    # 两者都认为必需的蛋白质
    both_essential = merged_df[
        (merged_df['prediction_human'] == 'Essential') &
        (merged_df['prediction_immune'] == 'Essential')
    ].copy()
    both_essential['avg_score'] = (both_essential['PES_score_human'] + both_essential['PES_score_immune']) / 2
    both_essential = both_essential.sort_values('avg_score', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    if len(both_essential) > 0:
        print(f"\nConsensus essential proteins (top 10):")
        print("=" * 80)
        for i, row in both_essential.head(10).iterrows():
            gene_name = row['gene_name_human']
            function = get_protein_function_annotation(row['protein_id'])
            print(f"  {i+1:2d}. {gene_name}")
            print(f"      功能: {function}")
            print(f"      人类PES: {row['PES_score_human']:.3f} | 免疫PES: {row['PES_score_immune']:.3f} | 平均: {row['avg_score']:.3f}")
            print()

    return human_specific, immune_specific, both_essential

def create_overview_visualization(merged_df):
    """创建概览可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Human vs Immune Cell Level Protein Essentiality Prediction Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 增强的PES分数散点图
    ax1 = axes[0, 0]

    # 根据预测一致性着色
    agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
    colors = [NATURE_COLORS['primary_blue'] if agree else NATURE_COLORS['primary_red']
              for agree in agreement]

    scatter = ax1.scatter(merged_df['PES_score_immune'], merged_df['PES_score_human'],
                         c=colors, alpha=0.6, s=25, edgecolors='white', linewidth=0.5)

    # 添加对角线和阈值线
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Agreement')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Human Threshold')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Immune Threshold')

    ax1.set_xlabel('Immune Cell Level PES Score', fontsize=12)
    ax1.set_ylabel('Human Level PES Score', fontsize=12)
    ax1.set_title('PES Score Correlation Analysis', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # 添加统计信息
    corr = stats.pearsonr(merged_df['PES_score_human'], merged_df['PES_score_immune'])[0]
    spearman_corr = stats.spearmanr(merged_df['PES_score_human'], merged_df['PES_score_immune'])[0]
    ax1.text(0.05, 0.95, f'Pearson r = {corr:.3f}\nSpearman ρ = {spearman_corr:.3f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 2. 增强的预测一致性分析
    ax2 = axes[0, 1]
    agreement_counts = [agreement.sum(), (~agreement).sum()]
    labels = ['Consistent\nPredictions', 'Inconsistent\nPredictions']
    colors = [NATURE_COLORS['primary_green'], NATURE_COLORS['primary_orange']]

    wedges, texts, autotexts = ax2.pie(agreement_counts, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11})
    ax2.set_title('Prediction Agreement Analysis', fontsize=13, fontweight='bold')

    # 添加数量信息
    for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
        autotext.set_text(f'{agreement_counts[i]}\n({autotext.get_text()})')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    # 3. 增强的PES分数分布对比
    ax3 = axes[1, 0]

    # 使用核密度估计创建更平滑的分布图
    from scipy.stats import gaussian_kde

    x_range = np.linspace(0, 1, 200)
    kde_human = gaussian_kde(merged_df['PES_score_human'])
    kde_immune = gaussian_kde(merged_df['PES_score_immune'])

    ax3.fill_between(x_range, kde_human(x_range), alpha=0.6,
                     color=NATURE_COLORS['primary_blue'], label='Human Level')
    ax3.fill_between(x_range, kde_immune(x_range), alpha=0.6,
                     color=NATURE_COLORS['primary_red'], label='Immune Cell Level')

    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
    ax3.set_xlabel('PES Score', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('PES Score Distribution Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # 4. 分数差异分析
    ax4 = axes[1, 1]
    score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']

    # 创建直方图
    n, bins, patches = ax4.hist(score_diff, bins=40, alpha=0.7,
                               color=NATURE_COLORS['primary_purple'], edgecolor='white')

    # 根据差异方向着色
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor(NATURE_COLORS['primary_red'])
        elif bins[i] > 0:
            patch.set_facecolor(NATURE_COLORS['primary_blue'])
        else:
            patch.set_facecolor(NATURE_COLORS['primary_purple'])

    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax4.axvline(x=score_diff.mean(), color='orange', linestyle='--', alpha=0.8,
                label=f'Mean = {score_diff.mean():.3f}')

    ax4.set_xlabel('PES Score Difference (Human - Immune)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Score Difference Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 添加统计信息
    ax4.text(0.05, 0.95, f'Mean: {score_diff.mean():.3f}\nStd: {score_diff.std():.3f}',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def create_detailed_comparison_analysis(merged_df, disagreement):
    """创建详细的对比分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Detailed Protein Essentiality Comparison Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 混淆矩阵热图
    ax1 = axes[0, 0]
    confusion_matrix = pd.crosstab(merged_df['prediction_human'],
                                  merged_df['prediction_immune'])

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Prediction Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Immune Cell Prediction', fontsize=12)
    ax1.set_ylabel('Human Level Prediction', fontsize=12)

    # 2. 不同预测组合的PES分数箱线图
    ax2 = axes[0, 1]

    # 基于高置信度阈值创建新的分组标签
    high_confidence_threshold = 0.8

    def get_confidence_group(row):
        if row['PES_score_human'] >= high_confidence_threshold and row['PES_score_immune'] >= high_confidence_threshold:
            return 'Both High-Conf'
        elif row['PES_score_human'] >= high_confidence_threshold and row['PES_score_immune'] < high_confidence_threshold:
            return 'Human High-Conf'
        elif row['PES_score_human'] < high_confidence_threshold and row['PES_score_immune'] >= high_confidence_threshold:
            return 'Immune High-Conf'
        else:
            return 'Both Low-Conf'

    merged_df['confidence_combo'] = merged_df.apply(get_confidence_group, axis=1)
    combo_order = ['Both High-Conf', 'Human High-Conf', 'Immune High-Conf', 'Both Low-Conf']

    # 过滤存在的组合
    existing_combos = [combo for combo in combo_order if combo in merged_df['confidence_combo'].values]

    if existing_combos:
        box_data = [merged_df[merged_df['confidence_combo'] == combo]['PES_score_human']
                   for combo in existing_combos]

        bp = ax2.boxplot(box_data, labels=[combo.replace('-', '-\n') for combo in existing_combos],
                        patch_artist=True)

        # 设置颜色
        colors = [NATURE_COLORS['primary_green'], NATURE_COLORS['primary_orange'],
                 NATURE_COLORS['primary_red'], NATURE_COLORS['primary_blue']]
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_title('Human PES Score by Confidence Group', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Human PES Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. 置信度分析
    ax3 = axes[0, 2]

    # 计算每种预测组合的置信度
    confidence_data = []
    confidence_labels = []

    for combo in existing_combos:
        combo_data = merged_df[merged_df['confidence_combo'] == combo]
        if len(combo_data) > 0:
            # 使用PES分数的标准差作为置信度的逆指标
            human_conf = 1 - combo_data['PES_score_human'].std()
            immune_conf = 1 - combo_data['PES_score_immune'].std()
            confidence_data.append([human_conf, immune_conf])
            confidence_labels.append(combo.replace('-', '-\n'))

    if confidence_data:
        confidence_array = np.array(confidence_data)
        x = np.arange(len(confidence_labels))
        width = 0.35

        ax3.bar(x - width/2, confidence_array[:, 0], width,
               label='Human Level', color=NATURE_COLORS['primary_blue'], alpha=0.8)
        ax3.bar(x + width/2, confidence_array[:, 1], width,
               label='Immune Level', color=NATURE_COLORS['primary_red'], alpha=0.8)

        ax3.set_xlabel('Confidence Group', fontsize=12)
        ax3.set_ylabel('Prediction Consistency', fontsize=12)
        ax3.set_title('Prediction Consistency Analysis', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(confidence_labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. 分歧蛋白质的特征分析
    ax4 = axes[1, 0]

    # 分析高置信度分组的蛋白质分布
    human_high_conf = merged_df[
        (merged_df['PES_score_human'] >= high_confidence_threshold) &
        (merged_df['PES_score_immune'] < high_confidence_threshold)
    ]
    immune_high_conf = merged_df[
        (merged_df['PES_score_human'] < high_confidence_threshold) &
        (merged_df['PES_score_immune'] >= high_confidence_threshold)
    ]

    if len(human_high_conf) > 0:
        ax4.scatter(human_high_conf['PES_score_immune'],
                   human_high_conf['PES_score_human'],
                   color=NATURE_COLORS['primary_blue'], alpha=0.7, s=30,
                   label=f'Human High-Confidence (n={len(human_high_conf)})')

    if len(immune_high_conf) > 0:
        ax4.scatter(immune_high_conf['PES_score_immune'],
                   immune_high_conf['PES_score_human'],
                   color=NATURE_COLORS['primary_red'], alpha=0.7, s=30,
                   label=f'Immune High-Confidence (n={len(immune_high_conf)})')

    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.axhline(y=high_confidence_threshold, color='orange', linestyle='--', alpha=0.8,
                label=f'High-Confidence Threshold ({high_confidence_threshold})')
    ax4.axvline(x=high_confidence_threshold, color='orange', linestyle='--', alpha=0.8)

    ax4.set_xlabel('Immune Cell PES Score', fontsize=12)
    ax4.set_ylabel('Human Level PES Score', fontsize=12)
    ax4.set_title('High-Confidence Proteins Analysis', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # 5. 分数差异的分位数分析
    ax5 = axes[1, 1]

    score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']

    # 计算分位数
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(score_diff, percentiles)

    # 创建分位数图
    colors_grad = [NATURE_COLORS['dark_red'], NATURE_COLORS['primary_red'],
                   NATURE_COLORS['primary_purple'], NATURE_COLORS['primary_blue'],
                   NATURE_COLORS['dark_blue']]

    bars = ax5.bar(range(len(percentiles)), percentile_values,
                   color=colors_grad, alpha=0.8, edgecolor='white')

    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax5.set_xlabel('Percentile', fontsize=12)
    ax5.set_ylabel('Score Difference (Human - Immune)', fontsize=12)
    ax5.set_title('Score Difference Percentile Analysis', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(percentiles)))
    ax5.set_xticklabels([f'{p}th' for p in percentiles])
    ax5.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, value in zip(bars, percentile_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    # 6. 功能富集分析（模拟）
    ax6 = axes[1, 2]

    # 基于蛋白质名称模拟功能分类
    def categorize_protein(protein_id):
        protein_id = protein_id.upper()
        if any(term in protein_id for term in ['RPS', 'RPL', 'RIBOSOM']):
            return 'Ribosomal'
        elif any(term in protein_id for term in ['H2B', 'H3', 'H4', 'HIST']):
            return 'Histone'
        elif any(term in protein_id for term in ['CREB', 'SOX', 'TLE', 'TRANSCR']):
            return 'Transcription'
        elif any(term in protein_id for term in ['ACSL', 'MME', 'METAB']):
            return 'Metabolism'
        else:
            return 'Other'

    if len(disagreement) > 0:
        disagreement['functional_category'] = disagreement['protein_id'].apply(categorize_protein)
        category_counts = disagreement['functional_category'].value_counts()

        if len(category_counts) > 0:
            wedges, texts, autotexts = ax6.pie(category_counts.values,
                                              labels=category_counts.index,
                                              autopct='%1.1f%%',
                                              colors=[NATURE_COLORS['primary_red'],
                                                     NATURE_COLORS['primary_blue'],
                                                     NATURE_COLORS['primary_green'],
                                                     NATURE_COLORS['primary_orange'],
                                                     NATURE_COLORS['primary_purple']][:len(category_counts)])
            ax6.set_title('Functional Categories of\nDisagreement Proteins',
                         fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig

def create_biomarker_analysis(merged_df, disagreement):
    """创建生物标志物分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Biomarker and High-Confidence Protein Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 定义高置信度阈值
    high_confidence_threshold = 0.8

    # 1. 高置信度蛋白质分布
    ax1 = axes[0, 0]

    human_high_conf = merged_df[merged_df['PES_score_human'] >= high_confidence_threshold]
    immune_high_conf = merged_df[merged_df['PES_score_immune'] >= high_confidence_threshold]
    both_high_conf = merged_df[(merged_df['PES_score_human'] >= high_confidence_threshold) &
                              (merged_df['PES_score_immune'] >= high_confidence_threshold)]

    categories = ['Human-only\nhigh-confidence', 'Immune-only\nhigh-confidence', 'Both models\nhigh-confidence']
    counts = [
        len(human_high_conf) - len(both_high_conf),
        len(immune_high_conf) - len(both_high_conf),
        len(both_high_conf)
    ]

    colors = [NATURE_COLORS['primary_blue'], NATURE_COLORS['primary_red'],
              NATURE_COLORS['primary_green']]

    bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='white')
    ax1.set_ylabel('Number of Proteins', fontsize=12)
    ax1.set_title(f'High-Confidence Proteins (PES ≥ {high_confidence_threshold})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. PES分数热图
    ax2 = axes[0, 1]

    # 选择前50个最有趣的蛋白质进行热图展示
    interesting_proteins = disagreement.head(50) if len(disagreement) > 50 else disagreement

    if len(interesting_proteins) > 0:
        heatmap_data = interesting_proteins[['PES_score_human', 'PES_score_immune']].T
        heatmap_data.columns = [f'P{i+1}' for i in range(len(interesting_proteins))]

        sns.heatmap(heatmap_data, cmap='RdYlBu_r', center=0.5,
                   cbar_kws={'label': 'PES Score'}, ax=ax2)
        ax2.set_title('PES Score Heatmap (Disagreement Proteins)',
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('Proteins (Top Disagreements)', fontsize=12)
        ax2.set_ylabel('Model Type', fontsize=12)
        ax2.set_yticklabels(['Human', 'Immune'], rotation=0)

    # 3. 分数差异vs平均分数散点图
    ax3 = axes[1, 0]

    score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']
    score_mean = (merged_df['PES_score_human'] + merged_df['PES_score_immune']) / 2

    # 根据一致性着色
    agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
    colors = [NATURE_COLORS['primary_green'] if agree else NATURE_COLORS['primary_orange']
              for agree in agreement]

    scatter = ax3.scatter(score_mean, score_diff, c=colors, alpha=0.6, s=25,
                         edgecolors='white', linewidth=0.5)

    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)

    ax3.set_xlabel('Average PES Score', fontsize=12)
    ax3.set_ylabel('Score Difference (Human - Immune)', fontsize=12)
    ax3.set_title('Bland-Altman Plot: Score Agreement Analysis',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 添加一致性界限
    mean_diff = score_diff.mean()
    std_diff = score_diff.std()
    ax3.axhline(y=mean_diff, color='red', linestyle='--', alpha=0.7,
                label=f'Mean Diff: {mean_diff:.3f}')
    ax3.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle=':', alpha=0.7,
                label=f'95% Limits')
    ax3.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle=':', alpha=0.7)
    ax3.legend(fontsize=10)

    # 4. 预测置信度分析
    ax4 = axes[1, 1]

    # 计算预测置信度（基于距离阈值的距离）
    human_confidence = np.abs(merged_df['PES_score_human'] - 0.5)
    immune_confidence = np.abs(merged_df['PES_score_immune'] - 0.5)

    # 创建置信度分布图
    ax4.hist(human_confidence, bins=30, alpha=0.7, label='Human Level',
            color=NATURE_COLORS['primary_blue'], density=True)
    ax4.hist(immune_confidence, bins=30, alpha=0.7, label='Immune Level',
            color=NATURE_COLORS['primary_red'], density=True)

    ax4.set_xlabel('Prediction Confidence (Distance from 0.5)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # 添加统计信息
    ax4.text(0.05, 0.95,
             f'Human Mean: {human_confidence.mean():.3f}\nImmune Mean: {immune_confidence.mean():.3f}',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def create_functional_analysis_visualization(merged_df, disagreement, human_specific, immune_specific, both_essential):
    """创建功能分析可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Functional Analysis of Protein Essentiality Differences',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 功能分类饼图 - 人类特异性蛋白质
    ax1 = axes[0, 0]
    if len(human_specific) > 0:
        human_specific_copy = human_specific.copy()
        human_specific_copy['function_category'] = human_specific_copy['protein_id'].apply(
            lambda x: get_protein_function_annotation(x).split(' - ')[0]
        )
        human_func_counts = human_specific_copy['function_category'].value_counts()

        colors = [NATURE_COLORS['primary_blue'], NATURE_COLORS['primary_green'],
                 NATURE_COLORS['primary_orange'], NATURE_COLORS['primary_purple'],
                 NATURE_COLORS['light_blue']][:len(human_func_counts)]

        # 将中文标签转换为英文
        english_labels = []
        for label in human_func_counts.index:
            if '转录' in label:
                english_labels.append('Transcription')
            elif '组蛋白' in label:
                english_labels.append('Histone')
            elif '代谢' in label:
                english_labels.append('Metabolism')
            elif '核糖体' in label:
                english_labels.append('Ribosomal')
            elif '表观遗传' in label:
                english_labels.append('Epigenetic')
            elif '细胞周期' in label:
                english_labels.append('Cell Cycle')
            else:
                english_labels.append('Other')

        wedges, texts, autotexts = ax1.pie(human_func_counts.values,
                                          labels=english_labels,
                                          autopct='%1.1f%%', colors=colors,
                                          textprops={'fontsize': 9})

        # 设置标签文字字体
        for text in texts:
            text.set_fontsize(9)

        # 设置百分比文字字体
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')

        ax1.set_title('Human High-Confidence Proteins\nFunctional Categories',
                     fontsize=12, fontweight='bold')

    # 2. 功能分类饼图 - 免疫特异性蛋白质
    ax2 = axes[0, 1]
    if len(immune_specific) > 0:
        immune_specific_copy = immune_specific.copy()
        immune_specific_copy['function_category'] = immune_specific_copy['protein_id'].apply(
            lambda x: get_protein_function_annotation(x).split(' - ')[0]
        )
        immune_func_counts = immune_specific_copy['function_category'].value_counts()

        colors = [NATURE_COLORS['primary_red'], NATURE_COLORS['primary_green'],
                 NATURE_COLORS['primary_orange'], NATURE_COLORS['primary_purple'],
                 NATURE_COLORS['light_red']][:len(immune_func_counts)]

        # 将中文标签转换为英文
        english_labels = []
        for label in immune_func_counts.index:
            if '转录' in label:
                english_labels.append('Transcription')
            elif '组蛋白' in label:
                english_labels.append('Histone')
            elif '代谢' in label:
                english_labels.append('Metabolism')
            elif '核糖体' in label:
                english_labels.append('Ribosomal')
            elif '表观遗传' in label:
                english_labels.append('Epigenetic')
            elif '细胞周期' in label:
                english_labels.append('Cell Cycle')
            else:
                english_labels.append('Other')

        wedges, texts, autotexts = ax2.pie(immune_func_counts.values,
                                          labels=english_labels,
                                          autopct='%1.1f%%', colors=colors,
                                          textprops={'fontsize': 9})

        # 设置标签文字字体
        for text in texts:
            text.set_fontsize(9)

        # 设置百分比文字字体
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')

        ax2.set_title('Immune High-Confidence Proteins\nFunctional Categories',
                     fontsize=12, fontweight='bold')

    # 3. 共识蛋白质功能分类
    ax3 = axes[0, 2]
    if len(both_essential) > 0:
        both_essential_copy = both_essential.copy()
        both_essential_copy['function_category'] = both_essential_copy['protein_id'].apply(
            lambda x: get_protein_function_annotation(x).split(' - ')[0]
        )
        both_func_counts = both_essential_copy['function_category'].value_counts()

        colors = [NATURE_COLORS['primary_green'], NATURE_COLORS['primary_blue'],
                 NATURE_COLORS['primary_orange'], NATURE_COLORS['primary_purple'],
                 NATURE_COLORS['light_green']][:len(both_func_counts)]

        # 将中文标签转换为英文
        english_labels = []
        for label in both_func_counts.index:
            if '转录' in label:
                english_labels.append('Transcription')
            elif '组蛋白' in label:
                english_labels.append('Histone')
            elif '代谢' in label:
                english_labels.append('Metabolism')
            elif '核糖体' in label:
                english_labels.append('Ribosomal')
            elif '表观遗传' in label:
                english_labels.append('Epigenetic')
            elif '细胞周期' in label:
                english_labels.append('Cell Cycle')
            else:
                english_labels.append('Other')

        wedges, texts, autotexts = ax3.pie(both_func_counts.values,
                                          labels=english_labels,
                                          autopct='%1.1f%%', colors=colors,
                                          textprops={'fontsize': 9})

        # 设置标签文字字体
        for text in texts:
            text.set_fontsize(9)

        # 设置百分比文字字体
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')

        ax3.set_title('Consensus Essential Proteins\nFunctional Categories',
                     fontsize=12, fontweight='bold')

    # 4. 顶级蛋白质详细信息 - 人类特异性
    ax4 = axes[1, 0]
    if len(human_specific) > 0:
        top_human = human_specific.head(8)
        y_pos = np.arange(len(top_human))
        scores = top_human['PES_score_human'].values

        bars = ax4.barh(y_pos, scores, color=NATURE_COLORS['primary_blue'], alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([name.split('-')[0] for name in top_human['gene_name_human']],
                           fontsize=10)
        ax4.set_xlabel('Human PES Score', fontsize=11)
        ax4.set_title('Top Human-Specific Proteins', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 添加分数标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax4.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)

    # 5. 顶级蛋白质详细信息 - 免疫特异性
    ax5 = axes[1, 1]
    if len(immune_specific) > 0:
        top_immune = immune_specific.head(8)
        y_pos = np.arange(len(top_immune))
        scores = top_immune['PES_score_immune'].values

        bars = ax5.barh(y_pos, scores, color=NATURE_COLORS['primary_red'], alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([name.split('-')[0] for name in top_immune['gene_name_immune']],
                           fontsize=10)
        ax5.set_xlabel('Immune PES Score', fontsize=11)
        ax5.set_title('Top Immune-Specific Proteins', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 添加分数标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax5.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)

    # 6. 生物学意义总结文本
    ax6 = axes[1, 2]
    ax6.axis('off')

    # 创建生物学意义总结 - 使用英文避免字体问题
    summary_text = """
Biological Significance Summary:

DNA Human-Specific Essential Proteins:
• Transcription factors (CREB5, SOX6)
• Epigenetic regulation (SMCHD1)
• General cell survival requirements

SHIELD Immune-Specific Essential Proteins:
• Histone-related (H2BC4, H2BC11)
• Cell cycle regulation (HYCC2)
• Rapid immune response activation

BALANCE Consensus Essential Proteins:
• Ribosomal proteins (RPS4Y2)
• Metabolic enzymes (NAMPT, SOD2)
• Basic cellular function maintenance

LIGHTBULB Clinical Significance:
• Human-specific: Broad therapeutic targets
• Immune-specific: Precision immune regulation
• Consensus proteins: Core survival mechanisms
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=NATURE_COLORS['light_blue'], alpha=0.3))

    plt.tight_layout()
    return fig

def create_physicochemical_analysis_visualization(merged_df, human_specific, immune_specific):
    """创建理化性质分析可视化图表"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Physicochemical Properties Analysis of Essential Proteins',
                 fontsize=16, fontweight='bold', y=0.98)

    # 定义理化性质和标签
    properties = ['molecular_weight', 'hydrophobicity', 'net_charge', 'isoelectric_point',
                  'polarity_ratio', 'aromatic_ratio', 'hydrophobic_ratio', 'charged_ratio']

    property_labels = {
        'molecular_weight': 'Molecular Weight (kDa)',
        'hydrophobicity': 'Hydrophobicity Index',
        'net_charge': 'Net Charge',
        'isoelectric_point': 'Isoelectric Point (pI)',
        'polarity_ratio': 'Polar AA (%)',
        'aromatic_ratio': 'Aromatic AA (%)',
        'hydrophobic_ratio': 'Hydrophobic AA (%)',
        'charged_ratio': 'Charged AA (%)'
    }

    # 前8个子图用于理化性质比较
    for i, prop in enumerate(properties):
        if i < 8:
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            # 使用带后缀的列名
            human_prop = f"{prop}_human"
            immune_prop = f"{prop}_immune"

            if human_prop in merged_df.columns or immune_prop in merged_df.columns:
                # 准备数据
                data_to_plot = []
                labels = []

                # 人类特异性 - 使用人类层面的数据
                if len(human_specific) > 0 and human_prop in human_specific.columns:
                    human_data = human_specific[human_prop].dropna()
                    if len(human_data) > 0:
                        data_to_plot.append(human_data)
                        labels.append(f'Human High-Conf\n(n={len(human_data)})')

                # 免疫特异性 - 使用免疫层面的数据
                if len(immune_specific) > 0 and immune_prop in immune_specific.columns:
                    immune_data = immune_specific[immune_prop].dropna()
                    if len(immune_data) > 0:
                        data_to_plot.append(immune_data)
                        labels.append(f'Immune High-Conf\n(n={len(immune_data)})')

                # 高置信度双方 - 使用人类层面的数据
                high_confidence_threshold = 0.8
                consensus_high_conf = merged_df[
                    (merged_df['PES_score_human'] >= high_confidence_threshold) &
                    (merged_df['PES_score_immune'] >= high_confidence_threshold)
                ]
                if len(consensus_high_conf) > 0 and human_prop in consensus_high_conf.columns:
                    consensus_data = consensus_high_conf[human_prop].dropna()
                    if len(consensus_data) > 0:
                        data_to_plot.append(consensus_data)
                        labels.append(f'Both High-Conf\n(n={len(consensus_data)})')

                if len(data_to_plot) > 0:
                    # 创建箱线图
                    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                    # 设置颜色
                    colors = [NATURE_COLORS['primary_blue'], NATURE_COLORS['primary_red'],
                             NATURE_COLORS['primary_green']]
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_title(property_labels.get(prop, prop), fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                    # 调整分子量单位
                    if prop == 'molecular_weight':
                        ax.set_ylabel('Molecular Weight (kDa)')
                        # 转换为kDa显示
                        ticks = ax.get_yticks()
                        ax.set_yticklabels([f'{tick/1000:.0f}' for tick in ticks])
                    else:
                        ax.set_ylabel(property_labels.get(prop, prop))

                    # 添加统计显著性
                    if len(data_to_plot) >= 2:
                        try:
                            stat, p_value = stats.ttest_ind(data_to_plot[0], data_to_plot[1])
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            ax.text(0.05, 0.95, f'p={p_value:.2e} {significance}',
                                   transform=ax.transAxes, fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        except:
                            pass

                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, f'{prop}\nData not available',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, style='italic')
                ax.set_xticks([])
                ax.set_yticks([])

    # 第9个子图：氨基酸组成热图比较
    ax9 = axes[2, 2]

    # 准备氨基酸组成数据
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    aa_data = []
    aa_labels = []

    # 计算人类特异性蛋白质的平均氨基酸组成
    if len(human_specific) > 0:
        human_aa_comp = []
        for aa in amino_acids:
            aa_percentages = []
            for _, row in human_specific.iterrows():
                aa_comp = row.get('aa_composition_human')
                if isinstance(aa_comp, dict):
                    aa_percentages.append(aa_comp.get(aa, 0))
            if aa_percentages:
                human_aa_comp.append(np.mean(aa_percentages))
            else:
                human_aa_comp.append(0)
        if any(x > 0 for x in human_aa_comp):  # 只有当有数据时才添加
            aa_data.append(human_aa_comp)
            aa_labels.append('Human High-Conf')

    # 计算免疫特异性蛋白质的平均氨基酸组成
    if len(immune_specific) > 0:
        immune_aa_comp = []
        for aa in amino_acids:
            aa_percentages = []
            for _, row in immune_specific.iterrows():
                aa_comp = row.get('aa_composition_immune')
                if isinstance(aa_comp, dict):
                    aa_percentages.append(aa_comp.get(aa, 0))
            if aa_percentages:
                immune_aa_comp.append(np.mean(aa_percentages))
            else:
                immune_aa_comp.append(0)
        if any(x > 0 for x in immune_aa_comp):  # 只有当有数据时才添加
            aa_data.append(immune_aa_comp)
            aa_labels.append('Immune High-Conf')

    if len(aa_data) > 0:
        print(f"DEBUG: Found amino acid data for {len(aa_data)} groups: {aa_labels}")
        print(f"DEBUG: Sample data - first group has {len(aa_data[0])} amino acids")
        print(f"DEBUG: Sample values: {aa_data[0][:5]}")  # 显示前5个氨基酸的值

        aa_array = np.array(aa_data)
        im = ax9.imshow(aa_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=15)

        ax9.set_xticks(range(len(amino_acids)))
        ax9.set_xticklabels(amino_acids, fontsize=10)
        ax9.set_yticks(range(len(aa_labels)))
        ax9.set_yticklabels(aa_labels, fontsize=10)
        ax9.set_title('Amino Acid Composition Comparison (%)', fontsize=11, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
        cbar.set_label('Percentage (%)', fontsize=10)

        # 只在数据不太多时添加数值标签
        if len(aa_data) <= 3 and len(amino_acids) <= 20:
            for i in range(len(aa_labels)):
                for j in range(len(amino_acids)):
                    if len(aa_data) > i and len(aa_data[i]) > j:
                        value = aa_data[i][j]
                        if value > 0.1:  # 只显示大于0.1%的值
                            text = ax9.text(j, i, f'{value:.1f}',
                                           ha="center", va="center",
                                           color="white" if value > 7.5 else "black",
                                           fontsize=7)
    else:
        print("DEBUG: No amino acid composition data found")
        ax9.text(0.5, 0.5, 'No amino acid\ncomposition data available',
                transform=ax9.transAxes, ha='center', va='center',
                fontsize=12, style='italic')
        ax9.set_xticks([])
        ax9.set_yticks([])

    plt.tight_layout()
    return fig

# create_disulfide_bond_analysis_visualization 函数已移至 disulfide_bond_analysis.py 模块

def generate_comprehensive_report(merged_df, disagreement, output_dir='../results/neutrophil_analysis'):
    """生成综合分析报告 - 增强版"""
    os.makedirs(output_dir, exist_ok=True)

    # 创建visualizations子目录
    viz_dir = f'{output_dir}/visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    print("Generating comprehensive visualization report...")

    # 获取特异性蛋白质数据 - 使用高置信度分组逻辑
    high_confidence_threshold = 0.8

    human_specific = merged_df[
        (merged_df['PES_score_human'] >= high_confidence_threshold) &
        (merged_df['PES_score_immune'] < high_confidence_threshold)
    ].sort_values('PES_score_human', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    immune_specific = merged_df[
        (merged_df['PES_score_human'] < high_confidence_threshold) &
        (merged_df['PES_score_immune'] >= high_confidence_threshold)
    ].sort_values('PES_score_immune', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    both_essential = merged_df[
        (merged_df['prediction_human'] == 'Essential') &
        (merged_df['prediction_immune'] == 'Essential')
    ].copy()
    both_essential['avg_score'] = (both_essential['PES_score_human'] + both_essential['PES_score_immune']) / 2
    both_essential = both_essential.sort_values('avg_score', ascending=False).drop_duplicates(subset='protein_id', keep='first')

    # 生成所有图表
    fig1 = create_overview_visualization(merged_df)
    fig1.savefig(f'{viz_dir}/01_overview_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Overview comparison saved")

    fig2 = create_detailed_comparison_analysis(merged_df, disagreement)
    fig2.savefig(f'{viz_dir}/02_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Detailed analysis saved")

    fig3 = create_biomarker_analysis(merged_df, disagreement)
    fig3.savefig(f'{viz_dir}/03_biomarker_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Biomarker analysis saved")

    # 新增：功能分析可视化
    fig4 = create_functional_analysis_visualization(merged_df, disagreement,
                                                   human_specific, immune_specific, both_essential)
    fig4.savefig(f'{viz_dir}/04_functional_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Functional analysis saved")

    # 新增：理化性质分析可视化
    fig5 = create_physicochemical_analysis_visualization(merged_df, human_specific, immune_specific)
    fig5.savefig(f'{viz_dir}/05_physicochemical_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Physicochemical analysis saved")

    # 二硫键分析可视化已移至独立模块
    # fig6 = create_disulfide_bond_analysis_visualization(merged_df, human_specific, immune_specific)
    # fig6.savefig(f'{viz_dir}/06_disulfide_analysis.png', dpi=300, bbox_inches='tight')
    # print("✓ Disulfide bond analysis saved")

    # 生成增强的文本报告
    generate_enhanced_text_report(merged_df, disagreement, output_dir,
                                 human_specific, immune_specific, both_essential)

    plt.close('all')
    print(f"\nAll visualizations saved to '{output_dir}' directory")

    return output_dir

def generate_enhanced_text_report(merged_df, disagreement, output_dir, human_specific=None, immune_specific=None, both_essential=None):
    """生成增强的文本分析报告"""
    report_path = f'{output_dir}/enhanced_comparison_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENHANCED HUMAN vs IMMUNE CELL LEVEL PROTEIN ESSENTIALITY COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        # 基本统计
        f.write("1. BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total proteins analyzed: {len(merged_df)}\n")

        human_essential = len(merged_df[merged_df['prediction_human'] == 'Essential'])
        immune_essential = len(merged_df[merged_df['prediction_immune'] == 'Essential'])

        f.write(f"Human level essential proteins: {human_essential} ({human_essential/len(merged_df)*100:.1f}%)\n")
        f.write(f"Immune level essential proteins: {immune_essential} ({immune_essential/len(merged_df)*100:.1f}%)\n\n")

        # 一致性分析
        agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
        agreement_rate = agreement.sum() / len(merged_df) * 100

        f.write("2. PREDICTION AGREEMENT ANALYSIS\n")
        f.write("-" * 32 + "\n")
        f.write(f"Consistent predictions: {agreement.sum()} ({agreement_rate:.1f}%)\n")
        f.write(f"Inconsistent predictions: {(~agreement).sum()} ({100-agreement_rate:.1f}%)\n\n")

        # 相关性分析
        corr_pearson = stats.pearsonr(merged_df['PES_score_human'], merged_df['PES_score_immune'])
        corr_spearman = stats.spearmanr(merged_df['PES_score_human'], merged_df['PES_score_immune'])

        f.write("3. CORRELATION ANALYSIS\n")
        f.write("-" * 23 + "\n")
        f.write(f"Pearson correlation: {corr_pearson[0]:.4f} (p-value: {corr_pearson[1]:.2e})\n")
        f.write(f"Spearman correlation: {corr_spearman[0]:.4f} (p-value: {corr_spearman[1]:.2e})\n\n")

        # 分数差异分析
        score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']
        f.write("4. SCORE DIFFERENCE ANALYSIS\n")
        f.write("-" * 28 + "\n")
        f.write(f"Mean difference (Human - Immune): {score_diff.mean():.4f}\n")
        f.write(f"Standard deviation: {score_diff.std():.4f}\n")
        f.write(f"Median difference: {score_diff.median():.4f}\n")
        f.write(f"Range: [{score_diff.min():.4f}, {score_diff.max():.4f}]\n\n")

        # 高置信度蛋白质分析
        high_conf_threshold = 0.8
        human_high_conf = merged_df[merged_df['PES_score_human'] >= high_conf_threshold]
        immune_high_conf = merged_df[merged_df['PES_score_immune'] >= high_conf_threshold]
        both_high_conf = merged_df[(merged_df['PES_score_human'] >= high_conf_threshold) &
                                  (merged_df['PES_score_immune'] >= high_conf_threshold)]

        f.write("5. HIGH-CONFIDENCE PROTEIN ANALYSIS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Threshold: PES ≥ {high_conf_threshold}\n")
        f.write(f"Human-only high-confidence: {len(human_high_conf) - len(both_high_conf)}\n")
        f.write(f"Immune-only high-confidence: {len(immune_high_conf) - len(both_high_conf)}\n")
        f.write(f"Both models high-confidence: {len(both_high_conf)}\n\n")

        # 不一致蛋白质详细分析
        if len(disagreement) > 0:
            f.write("6. DISAGREEMENT PROTEIN ANALYSIS\n")
            f.write("-" * 31 + "\n")

            human_essential_immune_non = disagreement[
                (disagreement['prediction_human'] == 'Essential') &
                (disagreement['prediction_immune'] == 'Non-essential')
            ].copy()
            human_essential_immune_non = human_essential_immune_non.drop_duplicates(subset='protein_id', keep='first')

            human_non_immune_essential = disagreement[
                (disagreement['prediction_human'] == 'Non-essential') &
                (disagreement['prediction_immune'] == 'Essential')
            ].copy()
            human_non_immune_essential = human_non_immune_essential.drop_duplicates(subset='protein_id', keep='first')

            f.write(f"Human Essential + Immune Non-essential: {len(human_essential_immune_non)}\n")
            f.write(f"Human Non-essential + Immune Essential: {len(human_non_immune_essential)}\n\n")

            # 顶级不一致蛋白质
            if len(human_essential_immune_non) > 0:
                f.write("Top 10 Human-specific Essential Proteins:\n")
                top_human_specific = human_essential_immune_non.nlargest(10, 'PES_score_human')
                for i, (_, row) in enumerate(top_human_specific.iterrows(), 1):
                    gene_name = row.get('gene_name_human', row['protein_id'][:30])
                    function = get_protein_function_annotation(row['protein_id'])
                    f.write(f"  {i:2d}. {gene_name}: Human={row['PES_score_human']:.3f}, Immune={row['PES_score_immune']:.3f}\n")
                    f.write(f"      功能: {function}\n")
                f.write("\n")

            if len(human_non_immune_essential) > 0:
                f.write("Top 10 Immune-specific Essential Proteins:\n")
                top_immune_specific = human_non_immune_essential.nlargest(10, 'PES_score_immune')
                for i, (_, row) in enumerate(top_immune_specific.iterrows(), 1):
                    gene_name = row.get('gene_name_immune', row['protein_id'][:30])
                    function = get_protein_function_annotation(row['protein_id'])
                    f.write(f"  {i:2d}. {gene_name}: Human={row['PES_score_human']:.3f}, Immune={row['PES_score_immune']:.3f}\n")
                    f.write(f"      功能: {function}\n")
                f.write("\n")

        # 一致的高分蛋白质
        both_essential = merged_df[
            (merged_df['prediction_human'] == 'Essential') &
            (merged_df['prediction_immune'] == 'Essential')
        ].copy()
        both_essential = both_essential.drop_duplicates(subset='protein_id', keep='first')

        if len(both_essential) > 0:
            f.write("7. CONSENSUS ESSENTIAL PROTEINS\n")
            f.write("-" * 29 + "\n")
            f.write(f"Proteins essential in both models: {len(both_essential)}\n\n")

            f.write("Top 10 Consensus Essential Proteins:\n")
            # 按两个分数的平均值排序
            both_essential = both_essential.assign(
                avg_score=(both_essential['PES_score_human'] + both_essential['PES_score_immune']) / 2
            )
            top_consensus = both_essential.nlargest(10, 'avg_score')
            for i, (_, row) in enumerate(top_consensus.iterrows(), 1):
                gene_name = row.get('gene_name_human', row['protein_id'][:30])
                f.write(f"  {i:2d}. {gene_name}: Human={row['PES_score_human']:.3f}, Immune={row['PES_score_immune']:.3f}, Avg={row['avg_score']:.3f}\n")
            f.write("\n")

        f.write("8. ANALYSIS SUMMARY\n")
        f.write("-" * 17 + "\n")
        f.write("Key Findings:\n")
        f.write(f"• Prediction agreement rate: {agreement_rate:.1f}%\n")
        f.write(f"• Score correlation (Pearson): {corr_pearson[0]:.3f}\n")
        f.write(f"• Mean score difference: {score_diff.mean():.3f}\n")
        f.write(f"• High-confidence proteins overlap: {len(both_high_conf)} proteins\n")
        f.write(f"• Total disagreement proteins: {len(disagreement)}\n\n")

        f.write("Biological Implications:\n")
        f.write("• Different models capture distinct aspects of protein essentiality\n")
        f.write("• Human-level model may reflect general cellular requirements\n")
        f.write("• Immune-level model captures immune-specific functional needs\n")
        f.write("• Combined analysis provides comprehensive essentiality assessment\n\n")

        # 添加详细的功能分析部分
        if human_specific is not None and len(human_specific) > 0:
            f.write("9. DETAILED FUNCTIONAL ANALYSIS\n")
            f.write("-" * 32 + "\n")

            # 人类特异性蛋白质功能分析
            f.write("Human-Specific Essential Proteins - Functional Patterns:\n")
            human_specific_copy = human_specific.copy()
            human_specific_copy['function_category'] = human_specific_copy['protein_id'].apply(
                lambda x: get_protein_function_annotation(x).split(' - ')[0]
            )
            human_func_counts = human_specific_copy['function_category'].value_counts()

            for func, count in human_func_counts.items():
                percentage = count / len(human_specific) * 100
                f.write(f"  • {func}: {count} proteins ({percentage:.1f}%)\n")

            f.write("\n生物学意义:\n")
            f.write("  - 转录调节因子在人类层面更重要，反映复杂的基因调控网络\n")
            f.write("  - 表观遗传调节蛋白质对维持细胞身份至关重要\n")
            f.write("  - 代谢相关蛋白质支持基础细胞生存需求\n\n")

        if immune_specific is not None and len(immune_specific) > 0:
            # 免疫特异性蛋白质功能分析
            f.write("Immune-Specific Essential Proteins - Functional Patterns:\n")
            immune_specific_copy = immune_specific.copy()
            immune_specific_copy['function_category'] = immune_specific_copy['protein_id'].apply(
                lambda x: get_protein_function_annotation(x).split(' - ')[0]
            )
            immune_func_counts = immune_specific_copy['function_category'].value_counts()

            for func, count in immune_func_counts.items():
                percentage = count / len(immune_specific) * 100
                f.write(f"  • {func}: {count} proteins ({percentage:.1f}%)\n")

            f.write("\n生物学意义:\n")
            f.write("  - 组蛋白相关蛋白质在免疫细胞激活中起关键作用\n")
            f.write("  - 细胞周期调节蛋白质支持快速增殖需求\n")
            f.write("  - 抗氧化酶应对免疫应答中的氧化应激\n\n")

        if both_essential is not None and len(both_essential) > 0:
            # 共识蛋白质功能分析
            f.write("Consensus Essential Proteins - Functional Patterns:\n")
            both_essential_copy = both_essential.copy()
            both_essential_copy['function_category'] = both_essential_copy['protein_id'].apply(
                lambda x: get_protein_function_annotation(x).split(' - ')[0]
            )
            both_func_counts = both_essential_copy['function_category'].value_counts()

            for func, count in both_func_counts.items():
                percentage = count / len(both_essential) * 100
                f.write(f"  • {func}: {count} proteins ({percentage:.1f}%)\n")

            f.write("\n生物学意义:\n")
            f.write("  - 核糖体蛋白质是所有细胞类型的基础需求\n")
            f.write("  - 代谢酶维持细胞能量和物质代谢\n")
            f.write("  - 这些蛋白质代表核心生存机制\n\n")

        f.write("10. CLINICAL AND RESEARCH IMPLICATIONS\n")
        f.write("-" * 37 + "\n")
        f.write("Drug Target Discovery:\n")
        f.write("  • Human-specific proteins: Broad-spectrum therapeutic targets\n")
        f.write("  • Immune-specific proteins: Precision immunomodulation targets\n")
        f.write("  • Consensus proteins: Core survival mechanisms (use with caution)\n\n")

        f.write("Disease Research Applications:\n")
        f.write("  • Autoimmune diseases: Focus on immune-specific essential proteins\n")
        f.write("  • Cancer research: Consider human-level essential proteins\n")
        f.write("  • Immunodeficiency: Investigate consensus essential proteins\n\n")

        f.write("Personalized Medicine:\n")
        f.write("  • Combined model predictions provide comprehensive risk assessment\n")
        f.write("  • Context-specific essentiality guides targeted interventions\n")
        f.write("  • Functional categories inform mechanism-based therapies\n\n")

        f.write("=" * 80 + "\n")
        f.write("ENHANCED REPORT WITH FUNCTIONAL ANALYSIS COMPLETE\n")
        f.write("Generated files: 01_overview_comparison.png, 02_detailed_analysis.png,\n")
        f.write("                03_biomarker_analysis.png, 04_functional_analysis.png\n")
        f.write("=" * 80 + "\n")

    print("✓ Enhanced text report saved")

def _serialize_aa_composition(value):
    """将氨基酸组成字典序列化为JSON字符串，便于导出"""
    if isinstance(value, dict) and value:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value if value not in ({}, []) else np.nan


def save_results(merged_df, disagreement, output_dir='../results/neutrophil_analysis'):
    """保存分析结果"""
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 保存完整的比较结果
    output_df = merged_df.copy()
    output_df['agreement'] = output_df['prediction_human'] == output_df['prediction_immune']
    output_df['score_difference'] = output_df['PES_score_human'] - output_df['PES_score_immune']

    for col in ['aa_composition_human', 'aa_composition_immune']:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(_serialize_aa_composition)

    comparison_path = os.path.join(data_dir, 'prediction_comparison_results.csv')
    output_df.to_csv(comparison_path, index=False)
    print(f"\nComplete comparison results saved as '{comparison_path}'")

    # 保存不一致的蛋白质
    if len(disagreement) > 0:
        disagreement_output = disagreement.copy()
        disagreement_output['score_difference'] = disagreement_output['PES_score_human'] - disagreement_output['PES_score_immune']
        for col in ['aa_composition_human', 'aa_composition_immune']:
            if col in disagreement_output.columns:
                disagreement_output[col] = disagreement_output[col].apply(_serialize_aa_composition)

        disagreement_path = os.path.join(data_dir, 'disagreement_proteins.csv')
        disagreement_output.to_csv(disagreement_path, index=False)
        print(f"Disagreement proteins saved as '{disagreement_path}'")

def main():
    """主函数 - 增强版本"""
    print("=" * 80)
    print("ENHANCED HUMAN vs IMMUNE CELL LEVEL PROTEIN ESSENTIALITY COMPARISON")
    print("=" * 80)

    try:
        # 加载数据
        print("\n📊 Loading and processing data...")
        human_df, immune_df = load_and_clean_data()

        # 对比预测结果
        print("🔍 Comparing predictions...")
        merged_df = compare_predictions(human_df, immune_df)

        # 分析一致性
        print("📈 Analyzing prediction agreement...")
        disagreement = analyze_agreement(merged_df)

        # 分析分数相关性
        print("📊 Analyzing score correlations...")
        analyze_score_correlation(merged_df)

        # 分析理化性质差异
        print("🧬 Analyzing physicochemical differences...")
        human_specific_phys, immune_specific_phys = analyze_physicochemical_differences(merged_df)

        # 二硫键分析已移至独立模块
        # print("🔗 Analyzing disulfide bond formation potential...")
        # analyze_disulfide_bonds(merged_df)

        # 识别有趣的蛋白质
        print("🎯 Identifying proteins of interest...")
        human_specific, immune_specific, both_essential = identify_interesting_proteins(merged_df, disagreement)

        # 生成综合可视化报告
        print("\n🎨 Generating enhanced visualizations...")
        output_dir = generate_comprehensive_report(merged_df, disagreement)

        # 保存原始结果（保持向后兼容）
        print("💾 Saving analysis results...")
        save_results(merged_df, disagreement, output_dir)

        print("\n" + "=" * 80)
        print("🎉 ENHANCED ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"📁 Enhanced visualizations saved to: {output_dir}/")
        print("📈 Generated files:")
        print("   • 01_overview_comparison.png - Comprehensive overview analysis")
        print("   • 02_detailed_analysis.png - Detailed comparison metrics")
        print("   • 03_biomarker_analysis.png - Biomarker and confidence analysis")
        print("   • 04_functional_analysis.png - Functional categories and biological insights")
        print("   • 05_physicochemical_analysis.png - Molecular physicochemical properties analysis")
        # print("   • 06_disulfide_analysis.png - Disulfide bond formation potential analysis")  # 已移至独立模块
        print("   • enhanced_comparison_report.txt - Detailed text report with functional analysis")
        print("   • prediction_comparison_results.csv - Complete comparison data")
        print("   • disagreement_proteins.csv - Proteins with inconsistent predictions")
        print("\n💡 Key Insights:")

        # 快速统计摘要
        agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
        agreement_rate = agreement.sum() / len(merged_df) * 100
        corr = stats.pearsonr(merged_df['PES_score_human'], merged_df['PES_score_immune'])[0]

        print(f"   • Prediction agreement: {agreement_rate:.1f}%")
        print(f"   • Score correlation: {corr:.3f}")
        print(f"   • Disagreement proteins: {len(disagreement)}")
        print(f"   • Total proteins analyzed: {len(merged_df)}")

        print("\n🔬 For detailed biological insights, see the enhanced_comparison_report.txt")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found!")
        print(f"   Please ensure the following files exist:")
        print(f"   • neutrophil_human_predictions.csv")
        print(f"   • neutrophil_immune_ensemble_predictions.csv")
        print(f"\n   Error details: {e}")

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        print(f"   Please check your data files and try again.")

if __name__ == "__main__":
    main()
