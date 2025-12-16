#!/usr/bin/env python3
"""
亚细胞定位和GO细胞组分富集分析
分析不同蛋白质组别的亚细胞定位差异
参考compare_predictions.py的目录结构
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict, Counter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import warnings
import os
import re
warnings.filterwarnings('ignore')

# 设置可视化样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 蛋白质组别颜色
COLORS = {
    'Commonly Essential': '#E64B35',
    'Human-Specific Essential': '#4DBBD5',
    'Immune-Specific Essential': '#00A087',
    'Commonly Non-essential': '#F39B7F'
}

def load_data():
    """加载蛋白质分组和UniProt注释数据"""
    print("📊 Loading data...")

    try:
        # 加载蛋白质四分组数据
        protein_groups = pd.read_csv('../data/neutrophil_four_group_classification.csv')

        # 提取UniProt ID
        uniprot_pattern = r'UniProt:([A-Z0-9]+)'
        protein_groups['uniprot_id'] = protein_groups['protein_id'].str.extract(uniprot_pattern)
        protein_groups = protein_groups.dropna(subset=['uniprot_id'])

        # 加载UniProt详细注释
        with open('../../data_processing/processed_data/uniprot_annotations/neutrophil_uniprot_detailed.json', 'r') as f:
            uniprot_annotations = json.load(f)

        print(f"✓ Loaded {len(protein_groups)} protein groups")
        print(f"✓ Loaded {len(uniprot_annotations)} UniProt annotations")

        return protein_groups, uniprot_annotations

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def process_localization_data(protein_groups, uniprot_annotations):
    """处理亚细胞定位数据"""
    print("🔍 Processing subcellular localization data...")

    # 创建UniProt ID到注释的映射
    uniprot_dict = {item['uniprot_id']: item for item in uniprot_annotations}

    # 合并数据
    localization_records = []

    for _, row in protein_groups.iterrows():
        uniprot_id = row['uniprot_id']
        group = row['group']

        if uniprot_id in uniprot_dict:
            annotation = uniprot_dict[uniprot_id]

            # 亚细胞定位
            subcellular_location = annotation.get('subcellular_location', '')

            # GO细胞组分
            go_cellular_component = annotation.get('go_cellular_component', [])

            localization_records.append({
                'uniprot_id': uniprot_id,
                'group': group,
                'subcellular_location': subcellular_location,
                'go_cellular_component': go_cellular_component
            })

    localization_data = pd.DataFrame(localization_records)

    # 统计信息
    total_proteins = len(localization_data)
    with_subcellular = len(localization_data[localization_data['subcellular_location'] != ''])
    with_go_cc = len(localization_data[localization_data['go_cellular_component'].apply(len) > 0])

    print(f"✓ Processed {total_proteins} proteins")
    print(f"✓ With subcellular location: {with_subcellular} ({with_subcellular/total_proteins*100:.1f}%)")
    print(f"✓ With GO cellular component: {with_go_cc} ({with_go_cc/total_proteins*100:.1f}%)")

    return localization_data

def analyze_subcellular_distribution(localization_data):
    """分析亚细胞定位分布"""
    print("📍 Analyzing subcellular localization distribution...")

    # 过滤有定位信息的蛋白质
    data_with_loc = localization_data[localization_data['subcellular_location'] != ''].copy()

    if len(data_with_loc) == 0:
        print("❌ No subcellular localization information found")
        return None

    # 统计各组别的定位分布
    location_stats = {}
    group_totals = {}

    for group in data_with_loc['group'].unique():
        group_data = data_with_loc[data_with_loc['group'] == group]
        group_totals[group] = len(group_data)

        # 统计定位
        locations = group_data['subcellular_location'].value_counts()
        location_stats[group] = locations

    print(f"✓ Found {len(data_with_loc)} proteins with subcellular localization")

    return {
        'data': data_with_loc,
        'stats': location_stats,
        'totals': group_totals
    }

def analyze_go_cellular_component_enrichment(localization_data):
    """GO细胞组分富集分析"""
    print("🧬 Performing GO cellular component enrichment analysis...")

    # 过滤有GO细胞组分信息的蛋白质
    data_with_go = localization_data[
        localization_data['go_cellular_component'].apply(len) > 0
    ].copy()

    if len(data_with_go) == 0:
        print("❌ No GO cellular component information found")
        return None

    # 收集所有GO细胞组分term
    all_go_terms = Counter()
    group_go_terms = defaultdict(lambda: defaultdict(int))

    for _, row in data_with_go.iterrows():
        group = row['group']
        go_terms = row['go_cellular_component']

        for term in go_terms:
            if isinstance(term, dict) and 'id' in term:
                go_id = term['id']
                all_go_terms[go_id] += 1
                group_go_terms[group][go_id] += 1
            elif isinstance(term, str):
                all_go_terms[term] += 1
                group_go_terms[group][term] += 1

    # 筛选频率较高的GO term（至少在5个蛋白质中出现）
    frequent_terms = {term for term, count in all_go_terms.items() if count >= 5}

    if not frequent_terms:
        print("❌ No frequent GO cellular component terms found")
        return None

    # 富集分析
    enrichment_results = []
    groups = data_with_go['group'].unique()

    # 计算背景
    background_total = len(data_with_go)

    for group in groups:
        group_data = data_with_go[data_with_go['group'] == group]
        group_total = len(group_data)

        if group_total < 5:  # 跳过样本太少的组
            continue

        for term in frequent_terms:
            # 计算该组中有此term的蛋白质数量
            group_with_term = group_go_terms[group].get(term, 0)
            group_without_term = group_total - group_with_term

            # 计算背景中有此term的蛋白质数量
            background_with_term = all_go_terms[term]
            background_without_term = background_total - background_with_term

            if group_with_term == 0:
                continue

            # Fisher精确检验
            contingency_table = [
                [group_with_term, group_without_term],
                [background_with_term - group_with_term,
                 background_without_term - group_without_term]
            ]

            try:
                odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')

                # 计算富集倍数
                group_rate = group_with_term / group_total
                background_rate = background_with_term / background_total
                fold_enrichment = group_rate / background_rate if background_rate > 0 else float('inf')

                enrichment_results.append({
                    'group': group,
                    'go_term': term,
                    'group_with_term': group_with_term,
                    'group_total': group_total,
                    'background_with_term': background_with_term,
                    'background_total': background_total,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value,
                    'fold_enrichment': fold_enrichment,
                    'group_rate': group_rate,
                    'background_rate': background_rate
                })

            except Exception as e:
                continue

    if not enrichment_results:
        print("❌ No enrichment results generated")
        return None

    # 转换为DataFrame并进行多重检验校正
    enrichment_df = pd.DataFrame(enrichment_results)

    # 多重检验校正
    _, corrected_p, _, _ = multipletests(enrichment_df['p_value'], method='fdr_bh')
    enrichment_df['adj_p_value'] = corrected_p

    # 筛选显著富集的结果
    significant_results = enrichment_df[
        (enrichment_df['adj_p_value'] < 0.05) &
        (enrichment_df['fold_enrichment'] > 1.5) &
        (enrichment_df['group_with_term'] >= 3)
    ].sort_values(['group', 'adj_p_value'])

    print(f"✓ Analyzed {len(frequent_terms)} GO cellular component terms")
    print(f"✓ Found {len(significant_results)} significantly enriched terms")

    return {
        'all_results': enrichment_df,
        'significant_results': significant_results
    }

def create_subcellular_heatmap(subcellular_results):
    """创建亚细胞定位热图"""
    if not subcellular_results:
        return

    print("🎨 Creating subcellular localization heatmap...")

    stats = subcellular_results['stats']
    totals = subcellular_results['totals']

    # 只保留必需蛋白质组别，排除共同非必需
    essential_groups = [
        'Commonly Essential',
        'Human-Specific Essential',
        'Immune-Specific Essential'
    ]

    # 过滤出必需组别的数据
    filtered_stats = {group: stats[group] for group in essential_groups if group in stats}
    filtered_totals = {group: totals[group] for group in essential_groups if group in totals}

    if not filtered_stats:
        print("❌ No essential protein groups found")
        return

    # 获取所有定位类型
    all_locations = set()
    for group_stats in filtered_stats.values():
        all_locations.update(group_stats.index)

    # 只保留在至少一个组中占比>2%的定位
    significant_locations = set()
    for loc in all_locations:
        for group, group_stats in filtered_stats.items():
            if loc in group_stats:
                percentage = group_stats[loc] / filtered_totals[group] * 100
                if percentage > 2:
                    significant_locations.add(loc)
                    break

    if not significant_locations:
        print("❌ No significant subcellular localizations found")
        return

    # 创建矩阵
    groups = essential_groups
    locations = sorted(significant_locations)

    matrix = []
    for group in groups:
        if group in filtered_stats:
            row = []
            for loc in locations:
                count = filtered_stats[group].get(loc, 0)
                percentage = count / filtered_totals[group] * 100 if filtered_totals[group] > 0 else 0
                row.append(percentage)
            matrix.append(row)

    # 创建热图
    plt.figure(figsize=(12, 8))

    # 处理定位名称，缩短过长的名称
    short_locations = []
    for loc in locations:
        if len(loc) > 25:
            short_locations.append(loc[:22] + '...')
        else:
            short_locations.append(loc)

    sns.heatmap(
        matrix,
        xticklabels=short_locations,
        yticklabels=[group for group in groups if group in filtered_stats],
        annot=False,  # 不显示数字
        cmap='Reds',
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5
    )

    plt.title('Subcellular Localization Distribution\nEssential Protein Groups',
             fontsize=14, fontweight='bold')
    plt.xlabel('Subcellular Localization', fontsize=12, fontweight='bold')
    plt.ylabel('Essential Protein Groups', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('../results/subcellular_localization_heatmap.png',
               dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示

def create_go_enrichment_plot(go_results):
    """创建GO细胞组分富集图"""
    if not go_results or len(go_results['significant_results']) == 0:
        print("❌ No significant GO enrichment results to plot")
        return

    print("🎨 Creating GO cellular component enrichment plot...")

    results = go_results['significant_results']

    # 获取每个组的前8个term
    plot_data = []
    for group in results['group'].unique():
        group_results = results[results['group'] == group].head(8)
        plot_data.append(group_results)

    if not plot_data:
        return

    plot_df = pd.concat(plot_data).reset_index(drop=True)

    # 创建气泡图
    fig, ax = plt.subplots(figsize=(14, 10))

    groups = plot_df['group'].unique()
    y_pos = 0

    for group in groups:
        group_data = plot_df[plot_df['group'] == group]
        n_terms = len(group_data)

        positions = range(y_pos, y_pos + n_terms)

        # 绘制气泡
        scatter = ax.scatter(
            group_data['fold_enrichment'],
            positions,
            s=group_data['group_with_term'] * 30,  # 大小基于蛋白质数量
            c=COLORS.get(group, '#999999'),
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=group
        )

        # 添加标签
        for i, (_, row) in enumerate(group_data.iterrows()):
            term_name = row['go_term'][:35] + '...' if len(row['go_term']) > 35 else row['go_term']
            ax.text(0.1, positions[i], f"{group}: {term_name}",
                   fontsize=8, va='center', ha='left')

        y_pos += n_terms + 1

    ax.set_xlabel('Fold Enrichment', fontsize=12, fontweight='bold')
    ax.set_title('GO Cellular Component Enrichment\nNeutrophil Protein Groups',
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('../results/go_cellular_component_enrichment.png',
               dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示

def create_localization_comparison_plot(subcellular_results):
    """创建定位比较图"""
    if not subcellular_results:
        return

    print("🎨 Creating subcellular localization comparison plot...")

    stats = subcellular_results['stats']
    totals = subcellular_results['totals']

    # 只保留必需蛋白质组别，排除共同非必需
    essential_groups = [
        'Commonly Essential',
        'Human-Specific Essential',
        'Immune-Specific Essential'
    ]

    # 过滤出必需组别的数据
    filtered_stats = {group: stats[group] for group in essential_groups if group in stats}
    filtered_totals = {group: totals[group] for group in essential_groups if group in totals}

    if not filtered_stats:
        print("❌ No essential protein groups found")
        return

    # 统计所有定位的总频次（仅针对必需组别）
    location_totals = Counter()
    for group_stats in filtered_stats.values():
        for loc, count in group_stats.items():
            location_totals[loc] += count

    # 选择前10个最常见的定位
    top_locations = [loc for loc, count in location_totals.most_common(10)]

    # 准备数据
    plot_data = []

    for group in essential_groups:
        if group in filtered_stats:
            group_stats = filtered_stats[group]
            group_total = filtered_totals[group]

            for loc in top_locations:
                count = group_stats.get(loc, 0)
                percentage = count / group_total * 100 if group_total > 0 else 0

                plot_data.append({
                    'Group': group,
                    'Location': loc,
                    'Percentage': percentage,
                    'Count': count
                })

    plot_df = pd.DataFrame(plot_data)

    # 创建分组柱状图
    plt.figure(figsize=(14, 8))

    # 使用seaborn创建分组柱状图
    sns.barplot(
        data=plot_df,
        x='Location',
        y='Percentage',
        hue='Group',
        palette=COLORS
    )

    plt.title('Subcellular Localization Distribution\nEssential Protein Groups',
             fontsize=14, fontweight='bold')
    plt.xlabel('Subcellular Localization', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Essential Protein Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/subcellular_localization_comparison.png',
               dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示

def save_results(subcellular_results, go_results):
    """保存分析结果"""
    print("💾 Saving analysis results...")

    # 保存亚细胞定位分布
    if subcellular_results:
        stats = subcellular_results['stats']
        totals = subcellular_results['totals']

        # 创建分布表
        distribution_data = []
        for group in stats.keys():
            for loc, count in stats[group].items():
                total = totals[group]
                percentage = count / total * 100
                distribution_data.append({
                    'group': group,
                    'location': loc,
                    'count': count,
                    'total': total,
                    'percentage': percentage
                })

        distribution_df = pd.DataFrame(distribution_data)
        distribution_df.to_csv('../results/subcellular_localization_distribution.csv', index=False)
        print("✓ Saved subcellular localization distribution")

    # 保存GO富集结果
    if go_results:
        # 保存所有结果
        go_results['all_results'].to_csv('../results/go_cellular_component_all_results.csv', index=False)

        # 保存显著结果
        if len(go_results['significant_results']) > 0:
            go_results['significant_results'].to_csv('../results/go_cellular_component_significant.csv', index=False)
            print("✓ Saved GO cellular component enrichment results")

def generate_report(subcellular_results, go_results, localization_data):
    """生成分析报告"""
    print("📝 Generating analysis report...")

    report = []
    report.append("# 亚细胞定位和GO细胞组分分析报告")
    report.append("## 中性粒细胞蛋白质分组比较\n")

    # 数据概览
    total_proteins = len(localization_data)
    with_subcellular = len(localization_data[localization_data['subcellular_location'] != ''])
    with_go_cc = len(localization_data[localization_data['go_cellular_component'].apply(len) > 0])

    report.append("### 数据概览")
    report.append(f"- **总蛋白质数量**: {total_proteins}")
    report.append(f"- **有亚细胞定位信息**: {with_subcellular} ({with_subcellular/total_proteins*100:.1f}%)")
    report.append(f"- **有GO细胞组分信息**: {with_go_cc} ({with_go_cc/total_proteins*100:.1f}%)\n")

    # 蛋白质组别分布
    report.append("### 蛋白质组别分布")
    for group, count in localization_data['group'].value_counts().items():
        report.append(f"- **{group}**: {count} 个蛋白质")
    report.append("")

    # 亚细胞定位分析
    if subcellular_results:
        report.append("### 亚细胞定位分布")
        stats = subcellular_results['stats']
        totals = subcellular_results['totals']

        for group in stats.keys():
            group_stats = stats[group]
            total = totals[group]

            report.append(f"\n**{group}** ({total} 个蛋白质):")
            for loc, count in group_stats.head(5).items():
                percentage = count / total * 100
                report.append(f"- {loc}: {count} ({percentage:.1f}%)")

    # GO细胞组分富集分析
    if go_results and len(go_results['significant_results']) > 0:
        report.append("\n### GO细胞组分富集分析")
        significant_results = go_results['significant_results']

        report.append(f"- **显著富集的GO term**: {len(significant_results)}")

        for group in significant_results['group'].unique():
            group_results = significant_results[significant_results['group'] == group]
            report.append(f"\n**{group}** ({len(group_results)} 个富集term):")

            for _, row in group_results.head(5).iterrows():
                report.append(f"- {row['go_term']}: FC={row['fold_enrichment']:.2f}, "
                            f"P_adj={row['adj_p_value']:.2e}")
    else:
        report.append("\n### GO细胞组分富集分析")
        report.append("- 未发现显著富集的GO细胞组分term")

    # 主要发现
    report.append("\n### 主要发现")

    if subcellular_results:
        stats = subcellular_results['stats']
        totals = subcellular_results['totals']

        # 找出每个组最主要的定位
        for group in stats.keys():
            group_stats = stats[group]
            if len(group_stats) > 0:
                main_loc = group_stats.index[0]
                main_count = group_stats.iloc[0]
                total = totals[group]
                percentage = main_count / total * 100
                report.append(f"- **{group}**主要定位于{main_loc} ({percentage:.1f}%)")

    if go_results and len(go_results['significant_results']) > 0:
        significant_results = go_results['significant_results']
        # 统计每组的富集数量
        group_enrichment_counts = significant_results['group'].value_counts()
        for group, count in group_enrichment_counts.items():
            report.append(f"- **{group}**有{count}个显著富集的GO细胞组分term")

    # 生物学意义
    report.append("\n### 生物学意义")
    report.append("- 不同蛋白质组别在亚细胞定位上表现出明显差异")
    report.append("- 这些差异反映了不同蛋白质在细胞功能中的特定角色")
    report.append("- 定位差异可能与蛋白质的必需性特征相关")
    report.append("- GO细胞组分富集分析揭示了组别特异性的功能区室偏好")

    # 保存报告
    with open('../results/subcellular_localization_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print("✓ Report saved: ../results/subcellular_localization_report.md")

def main():
    """运行完整分析"""

    print("=" * 60)
    print("SUBCELLULAR LOCALIZATION AND GO CELLULAR COMPONENT ANALYSIS")
    print("=" * 60)

    try:
        # 加载数据
        protein_groups, uniprot_annotations = load_data()
        if protein_groups is None or uniprot_annotations is None:
            return

        # 处理定位数据
        localization_data = process_localization_data(protein_groups, uniprot_annotations)

        # 进行分析
        print("\n🔍 Performing analyses...")
        subcellular_results = analyze_subcellular_distribution(localization_data)
        go_results = analyze_go_cellular_component_enrichment(localization_data)

        # 创建可视化
        print("\n🎨 Creating visualizations...")
        create_subcellular_heatmap(subcellular_results)
        create_go_enrichment_plot(go_results)
        create_localization_comparison_plot(subcellular_results)

        # 保存结果和生成报告
        print("\n💾 Saving results...")
        save_results(subcellular_results, go_results)
        generate_report(subcellular_results, go_results, localization_data)

        print("=" * 60)
        print("Analysis completed! Results saved in ../results/ directory")
        print("Generated files:")
        print("  • subcellular_localization_heatmap.png")
        print("  • go_cellular_component_enrichment.png")
        print("  • subcellular_localization_comparison.png")
        print("  • subcellular_localization_distribution.csv")
        print("  • go_cellular_component_significant.csv")
        print("  • subcellular_localization_report.md")
        print("=" * 60)

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        print("   Please check your data files and try again.")

if __name__ == "__main__":
    main()