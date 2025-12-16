#!/usr/bin/env python3
"""
蛋白质结构域富集分析
分析不同蛋白质组别在蛋白质结构域上的富集差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
import json
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
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
        annotations = {}

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

def extract_domain_info(annotations):
    """提取蛋白质结构域信息"""
    print("提取蛋白质结构域信息...")

    domain_data = {}
    domain_counter = Counter()

    # annotations是一个列表，每个元素是一个蛋白质的字典
    for protein_entry in annotations:
        if isinstance(protein_entry, dict):
            # 获取蛋白质ID，可能是uniprot_id
            protein_id = protein_entry.get('uniprot_id', protein_entry.get('protein_id', ''))

            # 检查domains字段
            if 'domains' in protein_entry and protein_entry['domains']:
                domains = []
                for domain in protein_entry['domains']:
                    if isinstance(domain, dict) and 'name' in domain:
                        domain_name = domain['name']
                        domains.append(domain_name)
                        domain_counter[domain_name] += 1
                    elif isinstance(domain, str):
                        domains.append(domain)
                        domain_counter[domain] += 1

                if domains:
                    domain_data[protein_id] = domains

    print(f"找到 {len(domain_data)} 个蛋白质有结构域信息")
    print(f"总共发现 {len(domain_counter)} 种不同结构域")

    # 显示最常见的结构域
    print("\n最常见的10个结构域:")
    for domain, count in domain_counter.most_common(10):
        print(f"  {domain}: {count} 个蛋白质")

    return domain_data, domain_counter

def extract_uniprot_id(protein_id):
    """从protein_id字符串中提取UniProt ID"""
    import re
    # 匹配模式: UniProt:P01266 或类似格式
    match = re.search(r'UniProt:([A-Z0-9]+)', protein_id)
    if match:
        return match.group(1)
    return None

def calculate_domain_enrichment(groups, domain_data, domain_counter, min_proteins=5):
    """计算结构域富集分析"""
    print(f"计算结构域富集分析 (最少 {min_proteins} 个蛋白质)...")

    # 只分析频率较高的结构域
    common_domains = [domain for domain, count in domain_counter.items() if count >= min_proteins]
    print(f"分析 {len(common_domains)} 个常见结构域")

    enrichment_results = []
    domain_distributions = {}

    # 只分析必需蛋白质组别
    essential_groups = ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential']

    for group_name in essential_groups:
        group_proteins = set(groups[group_name])
        # 提取UniProt ID并找到有结构域信息的蛋白质
        group_with_domains = []
        for protein_id in group_proteins:
            uniprot_id = extract_uniprot_id(protein_id)
            if uniprot_id and uniprot_id in domain_data:
                group_with_domains.append(uniprot_id)

        print(f"\n分析 {group_name}: {len(group_with_domains)} 个有结构域信息的蛋白质")

        group_distributions = {}

        for domain in common_domains:
            # 计算该组中有此结构域的蛋白质数量
            group_with_domain = sum(1 for uniprot_id in group_with_domains
                                  if domain in domain_data.get(uniprot_id, []))
            group_without_domain = len(group_with_domains) - group_with_domain

            # 计算背景中有此结构域的蛋白质数量
            total_with_domain = domain_counter[domain]
            total_proteins = len(domain_data)
            background_with_domain = total_with_domain - group_with_domain
            background_without_domain = total_proteins - len(group_with_domains) - background_with_domain

            # 构建列联表
            contingency_table = [
                [group_with_domain, group_without_domain],
                [background_with_domain, background_without_domain]
            ]

            # 进行Fisher精确检验
            try:
                if all(sum(row) > 0 for row in contingency_table) and all(sum(col) > 0 for col in zip(*contingency_table)):
                    _, p_value = fisher_exact(contingency_table)

                    # 计算富集倍数
                    expected = (total_with_domain * len(group_with_domains)) / total_proteins
                    fold_enrichment = group_with_domain / expected if expected > 0 else 0

                    # 计算百分比
                    percentage = (group_with_domain / len(group_with_domains)) * 100 if len(group_with_domains) > 0 else 0

                    enrichment_results.append({
                        'group': group_name,
                        'domain': domain,
                        'count': group_with_domain,
                        'total': len(group_with_domains),
                        'percentage': percentage,
                        'fold_enrichment': fold_enrichment,
                        'p_value': p_value
                    })

                    group_distributions[domain] = percentage

            except Exception as e:
                print(f"警告: 计算 {domain} 在 {group_name} 中的富集时出错: {e}")

        domain_distributions[group_name] = group_distributions

    return pd.DataFrame(enrichment_results), domain_distributions

def multiple_testing_correction(enrichment_df, method='fdr_bh'):
    """多重检验校正"""
    from statsmodels.stats.multitest import multipletests

    if len(enrichment_df) == 0:
        return enrichment_df

    _, corrected_p, _, _ = multipletests(enrichment_df['p_value'], method=method)
    enrichment_df['corrected_p_value'] = corrected_p
    enrichment_df['significant'] = corrected_p < 0.05

    return enrichment_df

def create_domain_visualizations(enrichment_df, domain_distributions):
    """创建结构域分析可视化图表"""
    print("创建结构域分析可视化图表...")

    # 设置图形样式
    plt.style.use('default')

    # 1. 显著富集结构域气泡图
    plt.figure(figsize=(14, 10))

    if len(enrichment_df) > 0:
        # 筛选显著富集的结果
        significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05].copy()

        if len(significant_df) > 0:
            # 按组分面
            groups = significant_df['group'].unique()
            fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 8))
            if len(groups) == 1:
                axes = [axes]

            for i, group in enumerate(groups):
                group_data = significant_df[significant_df['group'] == group]

                if len(group_data) > 0:
                    # 按富集倍数排序
                    group_data = group_data.sort_values('fold_enrichment', ascending=True)

                    # 创建气泡图
                    scatter = axes[i].scatter(
                        group_data['fold_enrichment'],
                        range(len(group_data)),
                        s=group_data['percentage'] * 10,  # 气泡大小代表百分比
                        c=-np.log10(group_data['corrected_p_value']),  # 颜色代表显著性
                        cmap='Reds',
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5
                    )

                    # 设置y轴标签
                    axes[i].set_yticks(range(len(group_data)))
                    axes[i].set_yticklabels(group_data['domain'], fontsize=10)

                    # 设置标签和标题
                    axes[i].set_xlabel('Fold Enrichment', fontsize=12)
                    axes[i].set_title(f'{group}\nSignificantly Enriched Domains', fontsize=14)
                    axes[i].grid(True, alpha=0.3)

                    # 添加富集倍数参考线
                    axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
                else:
                    axes[i].text(0.5, 0.5, 'No Significantly Enriched Domains',
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'{group}', fontsize=14)

            # 添加颜色条
            if len(significant_df) > 0:
                cbar = plt.colorbar(scatter, ax=axes, shrink=0.6)
                cbar.set_label('-log10(Adjusted P-value)', fontsize=12)

            plt.tight_layout()
            plt.savefig('../results/domain_enrichment_bubble.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"发现 {len(significant_df)} 个显著富集的结构域")
        else:
            print("未发现显著富集的结构域")

    # 2. 结构域分布热图
    plt.figure(figsize=(16, 8))

    if domain_distributions:
        # 准备热图数据
        groups = list(domain_distributions.keys())
        all_domains = set()
        for group_data in domain_distributions.values():
            all_domains.update(group_data.keys())

        # 只显示在至少一个组中百分比>2%的结构域
        filtered_domains = []
        for domain in all_domains:
            max_percentage = max(domain_distributions[group].get(domain, 0) for group in groups)
            if max_percentage > 2.0:
                filtered_domains.append(domain)

        if filtered_domains:
            # 创建矩阵
            matrix = []
            for group in groups:
                row = [domain_distributions[group].get(domain, 0) for domain in filtered_domains]
                matrix.append(row)

            matrix = np.array(matrix)

            # 创建热图
            plt.figure(figsize=(max(12, len(filtered_domains) * 0.5), 6))
            sns.heatmap(
                matrix,
                xticklabels=[domain[:30] + '...' if len(domain) > 30 else domain for domain in filtered_domains],
                yticklabels=groups,
                annot=False,
                cmap='Reds',
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5
            )

            plt.title('Protein Domain Distribution Heatmap', fontsize=16, pad=20)
            plt.xlabel('Protein Domains', fontsize=12)
            plt.ylabel('Protein Groups', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('../results/domain_distribution_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"生成包含 {len(filtered_domains)} 个结构域的分布热图")
        else:
            print("没有足够的结构域数据用于生成热图")

def generate_domain_report(enrichment_df, domain_distributions, groups):
    """生成结构域分析报告"""
    print("生成结构域分析报告...")

    report_lines = []
    report_lines.append("# 蛋白质结构域富集分析报告")
    report_lines.append("## 中性粒细胞蛋白质分组比较")
    report_lines.append("")

    # 数据概览
    total_proteins = sum(len(protein_list) for protein_list in groups.values())
    with_domains = len([p for group_proteins in groups.values()
                       for p in group_proteins
                       if any(p in domain_data for domain_data in [{}])])

    report_lines.append("### 数据概览")
    report_lines.append(f"- **总蛋白质数量**: {total_proteins}")
    report_lines.append("")

    # 蛋白质组别分布
    report_lines.append("### 蛋白质组别分布")
    for group, proteins in groups.items():
        report_lines.append(f"- **{group}**: {len(proteins)} 个蛋白质")
    report_lines.append("")

    # 显著富集的结构域
    if len(enrichment_df) > 0:
        significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05]

        report_lines.append("### 显著富集的结构域")
        if len(significant_df) > 0:
            for group in significant_df['group'].unique():
                group_data = significant_df[significant_df['group'] == group]
                report_lines.append(f"\n**{group}** ({len(group_data)} 个显著富集结构域):")

                # 按富集倍数排序，显示前10个
                top_domains = group_data.sort_values('fold_enrichment', ascending=False).head(10)
                for _, row in top_domains.iterrows():
                    report_lines.append(
                        f"- {row['domain']}: {row['count']}/{row['total']} ({row['percentage']:.1f}%), "
                        f"富集倍数={row['fold_enrichment']:.2f}, P={row['corrected_p_value']:.2e}"
                    )
        else:
            report_lines.append("- 未发现显著富集的结构域")
        report_lines.append("")

    # 结构域分布统计
    report_lines.append("### 结构域分布统计")
    essential_groups = ['Commonly Essential', 'Human-Specific Essential', 'Immune-Specific Essential']

    for group in essential_groups:
        if group in domain_distributions:
            group_data = domain_distributions[group]
            if group_data:
                report_lines.append(f"\n**{group}**:")
                # 显示前5个最高百分比的结构域
                sorted_domains = sorted(group_data.items(), key=lambda x: x[1], reverse=True)[:5]
                for domain, percentage in sorted_domains:
                    report_lines.append(f"- {domain}: {percentage:.1f}%")

    report_lines.append("")

    # 主要发现
    report_lines.append("### 主要发现")
    if len(enrichment_df) > 0:
        significant_df = enrichment_df[enrichment_df['corrected_p_value'] < 0.05]

        for group in essential_groups:
            group_significant = significant_df[significant_df['group'] == group]
            if len(group_significant) > 0:
                top_domain = group_significant.sort_values('fold_enrichment', ascending=False).iloc[0]
                report_lines.append(
                    f"- **{group}**主要富集于{top_domain['domain']} "
                    f"({top_domain['percentage']:.1f}%, 富集倍数{top_domain['fold_enrichment']:.2f})"
                )

    report_lines.append("")

    # 生物学意义
    report_lines.append("### 生物学意义")
    report_lines.append("- 不同蛋白质组别在结构域分布上表现出明显差异")
    report_lines.append("- 这些差异反映了不同蛋白质的功能特化")
    report_lines.append("- 结构域富集模式可能与蛋白质的必需性特征相关")
    report_lines.append("- 结构域分析为理解蛋白质功能提供了分子水平的见解")

    # 保存报告
    with open('../results/domain_enrichment_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print("结构域分析报告已保存到: ../results/domain_enrichment_report.md")

def main():
    """主函数"""
    print("开始蛋白质结构域富集分析...")

    # 加载数据
    human_pred, immune_pred, annotations = load_data()

    # 分组蛋白质
    groups, merged_data = classify_proteins(human_pred, immune_pred)

    # 提取结构域信息
    domain_data, domain_counter = extract_domain_info(annotations)

    if len(domain_data) == 0:
        print("错误: 未找到结构域信息，无法进行分析")
        return

    # 计算富集分析
    enrichment_df, domain_distributions = calculate_domain_enrichment(
        groups, domain_data, domain_counter, min_proteins=10
    )

    # 多重检验校正
    if len(enrichment_df) > 0:
        enrichment_df = multiple_testing_correction(enrichment_df)

        # 保存富集分析结果
        enrichment_df.to_csv('../results/domain_enrichment_results.csv', index=False)
        print(f"富集分析结果已保存: {len(enrichment_df)} 个结果")

    # 创建可视化
    create_domain_visualizations(enrichment_df, domain_distributions)

    # 生成报告
    generate_domain_report(enrichment_df, domain_distributions, groups)

    print("\n蛋白质结构域富集分析完成!")
    print("生成的文件:")
    print("- ../results/domain_enrichment_results.csv")
    print("- ../results/domain_enrichment_bubble.png")
    print("- ../results/domain_distribution_heatmap.png")
    print("- ../results/domain_enrichment_report.md")

if __name__ == "__main__":
    main()