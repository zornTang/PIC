#!/usr/bin/env python3
"""
比较人类层面和免疫细胞层面的必需蛋白预测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """加载和清理数据"""
    # 读取人类层面预测结果
    human_df = pd.read_csv('neutrophil_human_level_predictions.csv')
    
    # 读取免疫细胞层面预测结果
    immune_df = pd.read_csv('neutrophil_mane_proteins_predictions_ensemble.csv')
    
    # 提取蛋白质ID的基因名称部分用于匹配
    def extract_gene_name(protein_id):
        try:
            parts = protein_id.split('|')
            for part in parts:
                if not part.startswith('ENS') and not part.startswith('OTT') and len(part) > 2:
                    return part
            return parts[-1] if parts else protein_id
        except:
            return protein_id
    
    human_df['gene_name'] = human_df['protein_id'].apply(extract_gene_name)
    immune_df['gene_name'] = immune_df['protein_id'].apply(extract_gene_name)
    
    return human_df, immune_df

def compare_predictions(human_df, immune_df):
    """对比两种预测结果"""
    # 合并数据集
    merged_df = pd.merge(
        human_df[['protein_id', 'gene_name', 'PES_score', 'prediction', 'confidence']],
        immune_df[['protein_id', 'gene_name', 'PES_score', 'prediction', 'confidence']],
        on='protein_id',
        suffixes=('_human', '_immune'),
        how='inner'
    )
    
    print(f"Successfully matched proteins: {len(merged_df)}")
    
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

def identify_interesting_proteins(merged_df, disagreement):
    """识别有趣的蛋白质"""
    print("\n=== Proteins of Special Interest ===")
    
    # 人类层面必需但免疫细胞层面非必需
    human_specific = disagreement[
        (disagreement['prediction_human'] == 'Essential') & 
        (disagreement['prediction_immune'] == 'Non-essential')
    ].sort_values('PES_score_human', ascending=False)
    
    if len(human_specific) > 0:
        print(f"\nHuman-specific essential proteins (top 10):")
        for i, row in human_specific.head(10).iterrows():
            print(f"  {row['gene_name_human']}: Human PES={row['PES_score_human']:.3f}, Immune PES={row['PES_score_immune']:.3f}")
    
    # 免疫细胞层面必需但人类层面非必需
    immune_specific = disagreement[
        (disagreement['prediction_human'] == 'Non-essential') & 
        (disagreement['prediction_immune'] == 'Essential')
    ].sort_values('PES_score_immune', ascending=False)
    
    if len(immune_specific) > 0:
        print(f"\nImmune-specific essential proteins (top 10):")
        for i, row in immune_specific.head(10).iterrows():
            print(f"  {row['gene_name_immune']}: Human PES={row['PES_score_human']:.3f}, Immune PES={row['PES_score_immune']:.3f}")
    
    # 两者都认为必需的蛋白质
    both_essential = merged_df[
        (merged_df['prediction_human'] == 'Essential') & 
        (merged_df['prediction_immune'] == 'Essential')
    ].sort_values(['PES_score_human', 'PES_score_immune'], ascending=False)
    
    if len(both_essential) > 0:
        print(f"\nProteins essential in both models (top 10):")
        for i, row in both_essential.head(10).iterrows():
            print(f"  {row['gene_name_human']}: Human PES={row['PES_score_human']:.3f}, Immune PES={row['PES_score_immune']:.3f}")

def create_visualizations(merged_df):
    """创建可视化图表"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PES分数散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(merged_df['PES_score_immune'], merged_df['PES_score_human'], 
                         alpha=0.6, s=20)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax1.set_xlabel('Immune Cell Level PES Score')
    ax1.set_ylabel('Human Level PES Score')
    ax1.set_title('PES Score Correlation')
    ax1.grid(True, alpha=0.3)
    
    # 添加相关系数
    corr = stats.pearsonr(merged_df['PES_score_human'], merged_df['PES_score_immune'])[0]
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # 2. 预测一致性饼图
    ax2 = axes[0, 1]
    agreement = merged_df['prediction_human'] == merged_df['prediction_immune']
    agreement_counts = [agreement.sum(), (~agreement).sum()]
    labels = ['Agreement', 'Disagreement']
    colors = ['lightgreen', 'lightcoral']
    ax2.pie(agreement_counts, labels=labels, colors=colors, autopct='%1.1f%%')
    ax2.set_title('Prediction Agreement')
    
    # 3. PES分数分布
    ax3 = axes[1, 0]
    ax3.hist(merged_df['PES_score_human'], bins=30, alpha=0.7, label='Human Level', color='blue')
    ax3.hist(merged_df['PES_score_immune'], bins=30, alpha=0.7, label='Immune Cell Level', color='red')
    ax3.set_xlabel('PES Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('PES Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 分数差异分布
    ax4 = axes[1, 1]
    score_diff = merged_df['PES_score_human'] - merged_df['PES_score_immune']
    ax4.hist(score_diff, bins=30, alpha=0.7, color='purple')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('PES Score Difference (Human - Immune)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('PES Score Difference Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'prediction_comparison.png'")

def save_results(merged_df, disagreement):
    """保存分析结果"""
    # 保存完整的比较结果
    output_df = merged_df.copy()
    output_df['agreement'] = output_df['prediction_human'] == output_df['prediction_immune']
    output_df['score_difference'] = output_df['PES_score_human'] - output_df['PES_score_immune']
    
    output_df.to_csv('prediction_comparison_results.csv', index=False)
    print(f"\nComplete comparison results saved as 'prediction_comparison_results.csv'")
    
    # 保存不一致的蛋白质
    if len(disagreement) > 0:
        disagreement_output = disagreement.copy()
        disagreement_output['score_difference'] = disagreement_output['PES_score_human'] - disagreement_output['PES_score_immune']
        disagreement_output.to_csv('disagreement_proteins.csv', index=False)
        print(f"Disagreement proteins saved as 'disagreement_proteins.csv'")

def main():
    """主函数"""
    print("=== Human Level vs Immune Cell Level Essential Protein Prediction Comparison ===")
    
    # 加载数据
    human_df, immune_df = load_and_clean_data()
    
    # 对比预测结果
    merged_df = compare_predictions(human_df, immune_df)
    
    # 分析一致性
    disagreement = analyze_agreement(merged_df)
    
    # 分析分数相关性
    analyze_score_correlation(merged_df)
    
    # 识别有趣的蛋白质
    identify_interesting_proteins(merged_df, disagreement)
    
    # 创建可视化
    create_visualizations(merged_df)
    
    # 保存结果
    save_results(merged_df, disagreement)
    
    print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 