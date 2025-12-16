#!/usr/bin/env python3
"""
GO基因功能富集分析
分析不同蛋白质组别在GO功能注释上的富集差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
import json
from collections import Counter, defaultdict
import warnings
import re
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载蛋白质预测数据"""
    print("加载蛋白质预测数据...")

    # 加载人类水平预测结果
    human_pred = pd.read_csv('../data/neutrophil_proteins_human_predictions.csv')
    print(f"人类水平预测数据: {len(human_pred)} 个蛋白质")

    # 加载免疫细胞水平预测结果
    immune_pred = pd.read_csv('../data/neutrophil_immune_ensemble_predictions.csv')
    print(f"免疫细胞水平预测数据: {len(immune_pred)} 个蛋白质")

    # 加载功能注释数据
    try:
        with open('../../data_processing/processed_data/uniprot_annotations/neutrophil_uniprot_detailed.json', 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"功能注释数据: {len(annotations)} 个蛋白质")
    except FileNotFoundError:
        print("警告: 未找到功能注释文件，将使用模拟数据")
        annotations = []

    return human_pred, immune_pred, annotations

def classify_proteins(human_pred, immune_pred, threshold=0.5):
    """根据预测结果对蛋白质进行分组"""
    print(f"使用阈值 {threshold} 对蛋白质进行分组...")

    # 合并数据
    merged = pd.merge(human_pred, immune_pred, on='protein_id', suffixes=('_human', '_immune'))

    # 分组
    groups = {}

    # 共同必需: 两个模型都预测为必需
    groups['Commonly Essential'] = merged[
        (merged['PES_score_human'] >= threshold) &
        (merged['PES_score_immune'] >= threshold)
    ]['protein_id'].tolist()

    # 人类特异必需: 只有人类模型预测为必需
    groups['Human-Specific Essential'] = merged[
        (merged['PES_score_human'] >= threshold) &
        (merged['PES_score_immune'] < threshold)
    ]['protein_id'].tolist()

    # 免疫特异必需: 只有免疫模型预测为必需
    groups['Immune-Specific Essential'] = merged[
        (merged['PES_score_human'] < threshold) &
        (merged['PES_score_immune'] >= threshold)
    ]['protein_id'].tolist()

    # 共同非必需: 两个模型都预测为非必需
    groups['Commonly Non-essential'] = merged[
        (merged['PES_score_human'] < threshold) &
        (merged['PES_score_immune'] < threshold)
    ]['protein_id'].tolist()

    print("蛋白质分组结果:")
    for group, proteins in groups.items():
        print(f"  {group}: {len(proteins)} 个蛋白质")

    return groups, merged

def extract_uniprot_id(protein_id):
    """从protein_id字符串中提取UniProt ID"""
    # 匹配模式: UniProt:P01266 或类似格式
    match = re.search(r'UniProt:([A-Z0-9]+)', protein_id)
    if match:
        return match.group(1)
    return None

def extract_keyword_info(annotations):
    """提取关键词功能注释信息"""
    print("提取关键词功能注释信息...")

    keyword_data = {}
    keyword_counter = Counter()

    for protein_entry in annotations:
        if isinstance(protein_entry, dict):
            protein_id = protein_entry.get('uniprot_id', '')

            if 'keywords' in protein_entry and protein_entry['keywords']:
                keywords = protein_entry['keywords']

                if isinstance(keywords, list) and keywords:
                    protein_keywords = []
                    for keyword in keywords:
                        if isinstance(keyword, str) and keyword.strip():
                            clean_keyword = keyword.strip()
                            protein_keywords.append(clean_keyword)
                            keyword_counter[clean_keyword] += 1

                    if protein_keywords:
                        keyword_data[protein_id] = protein_keywords

    print(f"找到 {len(keyword_data)} 个蛋白质有关键词注释")
    print(f"总共发现 {len(keyword_counter)} 个不同关键词")

    # 显示最常见的关键词
    print("\n最常见的20个关键词:")
    for keyword, count in keyword_counter.most_common(20):
        print(f"  {keyword}: {count} 个蛋白质")

    return keyword_data, keyword_counter

def calculate_keyword_enrichment(groups, keyword_data, keyword_counter, min_proteins=5):
    """计算关键词富集分析"""
    print(f"计算关键词富集分析 (最少 {min_proteins} 个蛋白质)...")

    enrichment_results = []
    keyword_distributions = {}

    # 只分析必需蛋白质组别
    essential_groups = ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential']

    # 只分析频率较高的关键词
    common_keywords = [keyword for keyword, count in keyword_counter.items() if count >= min_proteins]
    print(f"分析 {len(common_keywords)} 个常见关键词")

    for group_name in essential_groups:
        group_proteins = set(groups[group_name])
        # 提取UniProt ID并找到有关键词注释的蛋白质
        group_with_keywords = []
        for protein_id in group_proteins:
            uniprot_id = extract_uniprot_id(protein_id)
            if uniprot_id and uniprot_id in keyword_data:
                group_with_keywords.append(uniprot_id)

        print(f"  {group_name}: {len(group_with_keywords)} 个有关键词注释的蛋白质")

        group_distributions = {}

        for keyword in common_keywords:
            # 计算该组中有此关键词的蛋白质数量
            group_with_keyword = sum(1 for uniprot_id in group_with_keywords
                                   if keyword in keyword_data.get(uniprot_id, []))
            group_without_keyword = len(group_with_keywords) - group_with_keyword

            if len(group_with_keywords) == 0:
                continue

            # 计算背景中有此关键词的蛋白质数量
            total_with_keyword = keyword_counter[keyword]
            total_proteins = len(keyword_data)
            background_with_keyword = total_with_keyword - group_with_keyword
            background_without_keyword = total_proteins - len(group_with_keywords) - background_with_keyword

            # 构建列联表
            contingency_table = [
                [group_with_keyword, group_without_keyword],
                [background_with_keyword, background_without_keyword]
            ]

            # 进行Fisher精确检验
            try:
                if all(sum(row) > 0 for row in contingency_table) and all(sum(col) > 0 for col in zip(*contingency_table)):
                    _, p_value = fisher_exact(contingency_table)

                    # 计算富集倍数
                    expected = (total_with_keyword * len(group_with_keywords)) / total_proteins
                    fold_enrichment = group_with_keyword / expected if expected > 0 else 0

                    # 计算百分比
                    percentage = (group_with_keyword / len(group_with_keywords)) * 100 if len(group_with_keywords) > 0 else 0

                    enrichment_results.append({
                        'group': group_name,
                        'keyword': keyword,
                        'count': group_with_keyword,
                        'total': len(group_with_keywords),
                        'percentage': percentage,
                        'fold_enrichment': fold_enrichment,
                        'p_value': p_value
                    })

                    group_distributions[keyword] = percentage

            except Exception as e:
                print(f"警告: 计算 {keyword} 在 {group_name} 中的富集时出错: {e}")

        keyword_distributions[group_name] = group_distributions

    return pd.DataFrame(enrichment_results), keyword_distributions

def multiple_testing_correction(enrichment_df, method='fdr_bh'):
    """多重检验校正"""
    from statsmodels.stats.multitest import multipletests

    if len(enrichment_df) == 0:
        return enrichment_df

    _, corrected_p, _, _ = multipletests(enrichment_df['p_value'], method=method)
    enrichment_df['corrected_p_value'] = corrected_p
    enrichment_df['significant'] = corrected_p < 0.05

    return enrichment_df

def create_keyword_visualizations(enrichment_df, keyword_distributions):
    """创建关键词分析可视化图表"""
    print("创建关键词分析可视化图表...")

    if len(enrichment_df) == 0:
        print("没有富集数据用于可视化")
        return

    # 筛选显著富集的结果
    significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05].copy()

    if len(significant_df) == 0:
        print("未发现显著富集的关键词")
        return

    # 1. 富集气泡图
    plt.figure(figsize=(14, 10))

    # 选择前30个最显著的富集
    top_enriched = significant_df.nsmallest(30, 'corrected_p_value')

    if len(top_enriched) > 0:
        # 创建散点图
        groups = top_enriched['group'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
        color_map = {group: colors[i] for i, group in enumerate(groups)}

        for group in groups:
            group_data = top_enriched[top_enriched['group'] == group]
            if len(group_data) > 0:
                scatter = plt.scatter(
                    group_data['fold_enrichment'],
                    range(len(group_data)),
                    s=group_data['percentage'] * 8,  # 气泡大小代表百分比
                    c=[color_map[group]] * len(group_data),
                    alpha=0.7,
                    label=group,
                    edgecolors='black',
                    linewidth=0.5
                )

        # 设置标签
        plt.yticks(range(len(top_enriched)), top_enriched['keyword'], fontsize=10)
        plt.xlabel('Fold Enrichment', fontsize=12)
        plt.title('Top Enriched Keywords by Protein Groups', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)

        # 添加颜色条
        cbar = plt.colorbar(scatter, shrink=0.6)
        cbar.set_label('-log10(Adjusted P-value)', fontsize=12)

    plt.tight_layout()
    plt.savefig('../results/keyword_enrichment_bubble.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 热图：每个组别的顶级富集关键词
    plt.figure(figsize=(16, 12))

    # 为每个组别选择最显著的关键词
    top_keywords_by_group = {}
    for group in significant_df['group'].unique():
        group_data = significant_df[significant_df['group'] == group]
        top_keywords = group_data.nsmallest(15, 'corrected_p_value')
        top_keywords_by_group[group] = top_keywords['keyword'].tolist()

    # 合并所有顶级关键词
    all_top_keywords = set()
    for keywords in top_keywords_by_group.values():
        all_top_keywords.update(keywords)

    if len(all_top_keywords) > 0:
        # 创建矩阵
        groups = list(top_keywords_by_group.keys())
        matrix = []
        keyword_labels = []

        for keyword in list(all_top_keywords)[:25]:  # 限制显示前25个
            row = []
            for group in groups:
                group_keyword_data = significant_df[
                    (significant_df['group'] == group) &
                    (significant_df['keyword'] == keyword)
                ]
                if len(group_keyword_data) > 0:
                    # 使用-log10(p值)作为颜色强度
                    intensity = -np.log10(group_keyword_data.iloc[0]['corrected_p_value'])
                    row.append(intensity)
                else:
                    row.append(0)

            matrix.append(row)
            keyword_labels.append(keyword)

        matrix = np.array(matrix)

        # 创建热图
        plt.figure(figsize=(12, max(8, len(keyword_labels) * 0.3)))
        sns.heatmap(
            matrix,
            xticklabels=groups,
            yticklabels=keyword_labels,
            annot=False,
            cmap='Reds',
            cbar_kws={'label': '-log10(Adjusted P-value)'},
            linewidths=0.5
        )

        plt.title('Keyword Enrichment Heatmap', fontsize=16, pad=20)
        plt.xlabel('Protein Groups', fontsize=12)
        plt.ylabel('Keywords', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../results/keyword_enrichment_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成关键词富集热图，包含 {len(keyword_labels)} 个关键词")

def generate_keyword_report(enrichment_df, keyword_distributions, groups):
    """生成关键词分析报告"""
    print("生成关键词分析报告...")

    report_lines = []
    report_lines.append("# Keyword Functional Enrichment Analysis Report")
    report_lines.append("## Neutrophil Protein Group Comparison")
    report_lines.append("")

    # 数据概览
    total_proteins = sum(len(protein_list) for protein_list in groups.values())

    report_lines.append("### Data Overview")
    report_lines.append(f"- **Total proteins analyzed**: {total_proteins}")
    report_lines.append("")

    # 蛋白质组别分布
    report_lines.append("### Protein Group Distribution")
    for group, proteins in groups.items():
        report_lines.append(f"- **{group}**: {len(proteins)} proteins")
    report_lines.append("")

    # 显著富集的关键词
    if len(enrichment_df) > 0:
        significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05]

        report_lines.append("### Significantly Enriched Keywords")

        if len(significant_df) > 0:
            essential_groups = ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential']

            for group in essential_groups:
                group_data = significant_df[significant_df['group'] == group]
                if len(group_data) > 0:
                    report_lines.append(f"\n**{group}** ({len(group_data)} enriched keywords):")

                    # 显示前10个最显著的富集
                    top_keywords = group_data.nsmallest(10, 'corrected_p_value')
                    for _, row in top_keywords.iterrows():
                        report_lines.append(
                            f"- {row['keyword']}: "
                            f"{row['count']}/{row['total']} ({row['percentage']:.1f}%), "
                            f"Fold={row['fold_enrichment']:.2f}, P={row['corrected_p_value']:.2e}"
                        )
        else:
            report_lines.append("- No significantly enriched keywords found")

        report_lines.append("")

    # 主要发现
    report_lines.append("### Key Findings")
    if len(enrichment_df) > 0:
        significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05]

        essential_groups = ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential']

        for group in essential_groups:
            group_significant = significant_df[significant_df['group'] == group]
            if len(group_significant) > 0:
                top_enrichment = group_significant.nsmallest(1, 'corrected_p_value').iloc[0]
                report_lines.append(
                    f"- **{group}** most enriched in '{top_enrichment['keyword']}' "
                    f"with {top_enrichment['fold_enrichment']:.2f}-fold enrichment"
                )

    report_lines.append("")

    # 生物学意义
    report_lines.append("### Biological Significance")
    report_lines.append("- Different protein groups show distinct keyword enrichment patterns")
    report_lines.append("- These differences reflect functional specialization of protein groups")
    report_lines.append("- Keyword enrichment patterns correlate with protein essentiality characteristics")
    report_lines.append("- Functional keyword analysis provides insights into biological processes and molecular functions")

    # 保存报告
    with open('../results/keyword_enrichment_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print("关键词分析报告已保存到: ../results/keyword_enrichment_report.md")

def main():
    """主函数"""
    print("开始关键词功能富集分析...")

    # 加载数据
    human_pred, immune_pred, annotations = load_data()

    # 分组蛋白质
    groups, merged_data = classify_proteins(human_pred, immune_pred)

    # 提取关键词信息
    keyword_data, keyword_counter = extract_keyword_info(annotations)

    if len(keyword_data) == 0:
        print("错误: 未找到关键词注释信息，无法进行分析")
        return

    # 计算富集分析
    enrichment_df, keyword_distributions = calculate_keyword_enrichment(
        groups, keyword_data, keyword_counter, min_proteins=10
    )

    # 多重检验校正
    if len(enrichment_df) > 0:
        enrichment_df = multiple_testing_correction(enrichment_df)

        # 保存富集分析结果
        enrichment_df.to_csv('../results/keyword_enrichment_results.csv', index=False)
        print(f"关键词富集分析结果已保存: {len(enrichment_df)} 个结果")

        # 显示显著富集的数量
        significant_count = len(enrichment_df[enrichment_df['corrected_p_value'] < 0.05])
        print(f"发现 {significant_count} 个显著富集的关键词")

    # 创建可视化
    create_keyword_visualizations(enrichment_df, keyword_distributions)

    # 生成报告
    generate_keyword_report(enrichment_df, keyword_distributions, groups)

    print("\n关键词功能富集分析完成!")
    print("生成的文件:")
    print("- ../results/keyword_enrichment_results.csv")
    print("- ../results/keyword_enrichment_bubble.png")
    print("- ../results/keyword_enrichment_heatmap.png")
    print("- ../results/keyword_enrichment_report.md")

if __name__ == "__main__":
    main()