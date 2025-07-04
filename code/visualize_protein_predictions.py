#!/usr/bin/env python3
"""
蛋白质重要性预测结果可视化分析脚本
对neutrophil_predictions_sorted.csv进行全面的数据可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # 使用默认样式

# Nature期刊配色方案
NATURE_COLORS = {
    'primary_blue': '#1f77b4',      # 深蓝色
    'primary_red': '#d62728',       # 深红色  
    'primary_green': '#2ca02c',     # 深绿色
    'primary_orange': '#ff7f0e',    # 橙色
    'primary_purple': '#9467bd',    # 紫色
    'primary_brown': '#8c564b',     # 棕色
    'primary_pink': '#e377c2',      # 粉色
    'primary_gray': '#7f7f7f',      # 灰色
    'primary_olive': '#bcbd22',     # 橄榄色
    'primary_cyan': '#17becf',      # 青色
    'light_blue': '#aec7e8',        # 浅蓝色
    'light_red': '#ffbb78',         # 浅红色
    'light_green': '#98df8a',       # 浅绿色
    'background': '#f8f9fa',        # 背景色
    'grid': '#e9ecef'               # 网格色
}

# Nature期刊常用色彩序列
NATURE_PALETTE = [
    NATURE_COLORS['primary_blue'],
    NATURE_COLORS['primary_red'], 
    NATURE_COLORS['primary_green'],
    NATURE_COLORS['primary_orange'],
    NATURE_COLORS['primary_purple'],
    NATURE_COLORS['primary_brown'],
    NATURE_COLORS['primary_pink'],
    NATURE_COLORS['primary_gray']
]

class ProteinVisualizationAnalyzer:
    def __init__(self, csv_file):
        """初始化分析器"""
        self.csv_file = csv_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        print(f"Loading data from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.df)} protein records")
        print(f"Columns: {list(self.df.columns)}")
        
        # 数据预处理
        self.df['PES_category'] = pd.cut(
            self.df['PES_score'], 
            bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
            labels=['Very Low (0-0.5)', 'Low (0.5-0.7)', 'Medium (0.7-0.85)', 'High (0.85-0.95)', 'Very High (0.95-1.0)']
        )
        
        # 序列长度分类
        self.df['length_category'] = pd.cut(
            self.df['sequence_length'],
            bins=[0, 200, 500, 1000, 2000, float('inf')],
            labels=['Short (<200)', 'Medium (200-500)', 'Long (500-1000)', 'Very Long (1000-2000)', 'Ultra Long (>2000)']
        )
        
    def create_overview_plot(self):
        """创建总览图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Protein Prediction Analysis Overview', fontsize=16, fontweight='bold')
        
        # 1. PES分数分布直方图
        axes[0, 0].hist(self.df['PES_score'], bins=50, alpha=0.7, color=NATURE_COLORS['light_blue'], 
                       edgecolor=NATURE_COLORS['primary_blue'], linewidth=0.5)
        axes[0, 0].axvline(self.df['PES_score'].mean(), color=NATURE_COLORS['primary_red'], linestyle='--', 
                          label=f'Mean: {self.df["PES_score"].mean():.3f}', linewidth=2)
        axes[0, 0].axvline(self.df['PES_score'].median(), color=NATURE_COLORS['primary_orange'], linestyle='--',
                          label=f'Median: {self.df["PES_score"].median():.3f}', linewidth=2)
        axes[0, 0].set_xlabel('PES Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('PES Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测结果饼图
        prediction_counts = self.df['prediction'].value_counts()
        colors = [NATURE_COLORS['primary_red'], NATURE_COLORS['primary_blue']]
        axes[0, 1].pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[0, 1].set_title('Prediction Distribution')
        
        # 3. 置信度分布
        confidence_counts = self.df['confidence'].value_counts()
        axes[0, 2].bar(confidence_counts.index, confidence_counts.values, 
                      color=[NATURE_COLORS['primary_red'], NATURE_COLORS['primary_green'], NATURE_COLORS['primary_blue']],
                      edgecolor='white', linewidth=1)
        axes[0, 2].set_xlabel('Confidence Level')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Confidence Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 序列长度分布
        axes[1, 0].hist(self.df['sequence_length'], bins=50, alpha=0.7, color=NATURE_COLORS['light_green'], 
                       edgecolor=NATURE_COLORS['primary_green'], linewidth=0.5)
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sequence Length Distribution')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. PES分数 vs 序列长度散点图
        scatter = axes[1, 1].scatter(self.df['sequence_length'], self.df['PES_score'], 
                                   alpha=0.6, c=self.df['PES_score'], cmap='Blues', s=20,
                                   edgecolors=NATURE_COLORS['primary_blue'], linewidth=0.3)
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('PES Score')
        axes[1, 1].set_title('PES Score vs Sequence Length')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='PES Score')
        
        # 6. PES分类分布
        pes_cat_counts = self.df['PES_category'].value_counts().sort_index()
        nature_gradient = [NATURE_COLORS['light_blue'], NATURE_COLORS['primary_blue'], 
                          NATURE_COLORS['primary_green'], NATURE_COLORS['primary_orange'], NATURE_COLORS['primary_red']]
        axes[1, 2].bar(range(len(pes_cat_counts)), pes_cat_counts.values, 
                      color=nature_gradient[:len(pes_cat_counts)], edgecolor='white', linewidth=1)
        axes[1, 2].set_xticks(range(len(pes_cat_counts)))
        axes[1, 2].set_xticklabels(pes_cat_counts.index, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('PES Score Categories')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_detailed_analysis(self):
        """创建详细分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Protein Analysis', fontsize=16, fontweight='bold')
        
        # 1. 箱线图：不同预测结果的PES分数分布
        sns.boxplot(data=self.df, x='prediction', y='PES_score', ax=axes[0, 0],
                   palette=[NATURE_COLORS['primary_red'], NATURE_COLORS['primary_blue']])
        axes[0, 0].set_title('PES Score Distribution by Prediction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 小提琴图：不同置信度的PES分数分布
        sns.violinplot(data=self.df, x='confidence', y='PES_score', ax=axes[0, 1],
                      palette=[NATURE_COLORS['primary_red'], NATURE_COLORS['primary_green'], NATURE_COLORS['primary_blue']])
        axes[0, 1].set_title('PES Score Distribution by Confidence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 热力图：序列长度类别 vs PES类别
        heatmap_data = pd.crosstab(self.df['length_category'], self.df['PES_category'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='white')
        axes[1, 0].set_title('Sequence Length vs PES Category Heatmap')
        axes[1, 0].set_xlabel('PES Category')
        axes[1, 0].set_ylabel('Length Category')
        
        # 4. 累积分布图
        sorted_pes = np.sort(self.df['PES_score'])
        cumulative = np.arange(1, len(sorted_pes) + 1) / len(sorted_pes)
        axes[1, 1].plot(sorted_pes, cumulative, linewidth=3, color=NATURE_COLORS['primary_blue'])
        axes[1, 1].axhline(y=0.5, color=NATURE_COLORS['primary_red'], linestyle='--', alpha=0.8, 
                          label='50th percentile', linewidth=2)
        axes[1, 1].axhline(y=0.9, color=NATURE_COLORS['primary_orange'], linestyle='--', alpha=0.8, 
                          label='90th percentile', linewidth=2)
        axes[1, 1].axhline(y=0.95, color=NATURE_COLORS['primary_green'], linestyle='--', alpha=0.8, 
                          label='95th percentile', linewidth=2)
        axes[1, 1].set_xlabel('PES Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of PES Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_biomarker_analysis(self):
        """创建生物标志物分析图表"""
        # 定义生物标志物阈值
        biomarker_threshold = 0.9
        biomarkers = self.df[self.df['PES_score'] >= biomarker_threshold].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Biomarker Analysis (PES ≥ {biomarker_threshold})', fontsize=16, fontweight='bold')
        
        # 1. 生物标志物数量和分布
        total_proteins = len(self.df)
        biomarker_count = len(biomarkers)
        non_biomarker_count = total_proteins - biomarker_count
        
        labels = ['Potential Biomarkers', 'Non-Biomarkers']
        sizes = [biomarker_count, non_biomarker_count]
        colors = [NATURE_COLORS['primary_red'], NATURE_COLORS['light_blue']]
        
        axes[0, 0].pie(sizes, labels=labels, autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_proteins)})',
                      colors=colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[0, 0].set_title(f'Biomarker Distribution\n(Threshold: PES ≥ {biomarker_threshold})')
        
        # 2. 生物标志物的序列长度分布
        if len(biomarkers) > 0:
            axes[0, 1].hist(biomarkers['sequence_length'], bins=20, alpha=0.8, color=NATURE_COLORS['primary_red'], 
                           label=f'Biomarkers (n={len(biomarkers)})', edgecolor='white', linewidth=1)
            axes[0, 1].hist(self.df['sequence_length'], bins=20, alpha=0.6, color=NATURE_COLORS['light_blue'],
                           label=f'All proteins (n={len(self.df)})', edgecolor='white', linewidth=1)
            axes[0, 1].set_xlabel('Sequence Length')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sequence Length: Biomarkers vs All Proteins')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 生物标志物的PES分数分布
        if len(biomarkers) > 0:
            axes[1, 0].hist(biomarkers['PES_score'], bins=20, alpha=0.8, color=NATURE_COLORS['primary_green'], 
                           edgecolor='white', linewidth=1)
            axes[1, 0].axvline(biomarkers['PES_score'].mean(), color=NATURE_COLORS['primary_red'], linestyle='--',
                              label=f'Mean: {biomarkers["PES_score"].mean():.3f}', linewidth=2)
            axes[1, 0].set_xlabel('PES Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('PES Score Distribution of Biomarkers')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 顶级生物标志物表格
        if len(biomarkers) > 0:
            top_biomarkers = biomarkers.nlargest(10, 'PES_score')[['protein_id', 'PES_score', 'sequence_length']]
            
            # 创建表格
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            
            table_data = []
            for idx, row in top_biomarkers.iterrows():
                protein_id = row['protein_id'][:20] + '...' if len(row['protein_id']) > 20 else row['protein_id']
                table_data.append([protein_id, f"{row['PES_score']:.4f}", f"{row['sequence_length']}"])
            
            table = axes[1, 1].table(cellText=table_data,
                                   colLabels=['Protein ID', 'PES Score', 'Length'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # 设置表格样式
            for i in range(len(table_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # 标题行
                        cell.set_facecolor(NATURE_COLORS['primary_green'])
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor(NATURE_COLORS['background'] if i % 2 == 0 else 'white')
            
            axes[1, 1].set_title('Top 10 Potential Biomarkers')
        
        plt.tight_layout()
        return fig
    
    def create_statistical_summary(self):
        """创建统计摘要图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Summary', fontsize=16, fontweight='bold')
        
        # 1. 基本统计信息表格
        stats_data = {
            'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'PES Score': [
                len(self.df),
                f"{self.df['PES_score'].mean():.4f}",
                f"{self.df['PES_score'].std():.4f}",
                f"{self.df['PES_score'].min():.4f}",
                f"{self.df['PES_score'].quantile(0.25):.4f}",
                f"{self.df['PES_score'].quantile(0.5):.4f}",
                f"{self.df['PES_score'].quantile(0.75):.4f}",
                f"{self.df['PES_score'].max():.4f}"
            ],
            'Sequence Length': [
                len(self.df),
                f"{self.df['sequence_length'].mean():.1f}",
                f"{self.df['sequence_length'].std():.1f}",
                f"{self.df['sequence_length'].min()}",
                f"{self.df['sequence_length'].quantile(0.25):.1f}",
                f"{self.df['sequence_length'].quantile(0.5):.1f}",
                f"{self.df['sequence_length'].quantile(0.75):.1f}",
                f"{self.df['sequence_length'].max()}"
            ]
        }
        
        axes[0, 0].axis('tight')
        axes[0, 0].axis('off')
        table = axes[0, 0].table(cellText=[[stats_data['Metric'][i], 
                                          stats_data['PES Score'][i], 
                                          stats_data['Sequence Length'][i]] for i in range(8)],
                               colLabels=['Metric', 'PES Score', 'Sequence Length'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(9):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor(NATURE_COLORS['primary_blue'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(NATURE_COLORS['background'] if i % 2 == 0 else 'white')
        
        axes[0, 0].set_title('Descriptive Statistics')
        
        # 2. Q-Q图检验正态性
        from scipy import stats
        stats.probplot(self.df['PES_score'], dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot: PES Score Normality Test')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相关性分析
        correlation = self.df['PES_score'].corr(self.df['sequence_length'])
        axes[1, 0].scatter(self.df['sequence_length'], self.df['PES_score'], alpha=0.6, s=20,
                          color=NATURE_COLORS['primary_blue'], edgecolors=NATURE_COLORS['primary_blue'], linewidth=0.3)
        
        # 添加趋势线
        z = np.polyfit(self.df['sequence_length'], self.df['PES_score'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.df['sequence_length'], p(self.df['sequence_length']), 
                       color=NATURE_COLORS['primary_red'], linestyle='--', alpha=0.8, linewidth=3)
        
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('PES Score')
        axes[1, 0].set_title(f'Correlation Analysis\n(r = {correlation:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 分类统计
        category_stats = self.df.groupby('prediction').agg({
            'PES_score': ['count', 'mean', 'std'],
            'sequence_length': ['mean', 'std']
        }).round(3)
        
        # 创建分类统计表格
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # 准备表格数据
        table_data = []
        for pred in category_stats.index:
            row = [
                pred,
                f"{category_stats.loc[pred, ('PES_score', 'count')]}",
                f"{category_stats.loc[pred, ('PES_score', 'mean')]:.4f}",
                f"{category_stats.loc[pred, ('PES_score', 'std')]:.4f}",
                f"{category_stats.loc[pred, ('sequence_length', 'mean')]:.1f}"
            ]
            table_data.append(row)
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Prediction', 'Count', 'PES Mean', 'PES Std', 'Length Mean'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor(NATURE_COLORS['primary_orange'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(NATURE_COLORS['background'] if i % 2 == 0 else 'white')
        
        axes[1, 1].set_title('Statistics by Prediction Category')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_dir='neutrophil_analysis_results'):
        """生成完整的可视化报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualization report...")
        
        # 生成所有图表
        fig1 = self.create_overview_plot()
        fig1.savefig(f'{output_dir}/01_overview_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Overview analysis saved")
        
        fig2 = self.create_detailed_analysis()
        fig2.savefig(f'{output_dir}/02_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Detailed analysis saved")
        
        fig3 = self.create_biomarker_analysis()
        fig3.savefig(f'{output_dir}/03_biomarker_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Biomarker analysis saved")
        
        fig4 = self.create_statistical_summary()
        fig4.savefig(f'{output_dir}/04_statistical_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Statistical summary saved")
        
        # 生成数据摘要报告
        self.generate_text_report(output_dir)
        
        plt.close('all')
        print(f"\nAll visualizations saved to '{output_dir}' directory")
        
        return output_dir
    
    def generate_text_report(self, output_dir):
        """生成文本摘要报告"""
        report_path = f'{output_dir}/protein_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PROTEIN PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total proteins analyzed: {len(self.df)}\n")
            f.write(f"Data source: {self.csv_file}\n\n")
            
            # PES分数统计
            f.write("2. PES SCORE STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Mean PES score: {self.df['PES_score'].mean():.4f}\n")
            f.write(f"Median PES score: {self.df['PES_score'].median():.4f}\n")
            f.write(f"Standard deviation: {self.df['PES_score'].std():.4f}\n")
            f.write(f"Min PES score: {self.df['PES_score'].min():.4f}\n")
            f.write(f"Max PES score: {self.df['PES_score'].max():.4f}\n\n")
            
            # 预测结果分布
            f.write("3. PREDICTION DISTRIBUTION\n")
            f.write("-" * 27 + "\n")
            pred_counts = self.df['prediction'].value_counts()
            for pred, count in pred_counts.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"{pred}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 置信度分布
            f.write("4. CONFIDENCE DISTRIBUTION\n")
            f.write("-" * 28 + "\n")
            conf_counts = self.df['confidence'].value_counts()
            for conf, count in conf_counts.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"{conf}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 生物标志物分析
            f.write("5. BIOMARKER ANALYSIS\n")
            f.write("-" * 22 + "\n")
            biomarker_threshold = 0.9
            biomarkers = self.df[self.df['PES_score'] >= biomarker_threshold]
            f.write(f"Potential biomarkers (PES ≥ {biomarker_threshold}): {len(biomarkers)}\n")
            f.write(f"Biomarker percentage: {(len(biomarkers)/len(self.df)*100):.1f}%\n")
            
            if len(biomarkers) > 0:
                f.write(f"Average PES score of biomarkers: {biomarkers['PES_score'].mean():.4f}\n")
                f.write(f"Average sequence length of biomarkers: {biomarkers['sequence_length'].mean():.1f}\n")
            f.write("\n")
            
            # 序列长度统计
            f.write("6. SEQUENCE LENGTH STATISTICS\n")
            f.write("-" * 31 + "\n")
            f.write(f"Mean length: {self.df['sequence_length'].mean():.1f}\n")
            f.write(f"Median length: {self.df['sequence_length'].median():.1f}\n")
            f.write(f"Min length: {self.df['sequence_length'].min()}\n")
            f.write(f"Max length: {self.df['sequence_length'].max()}\n\n")
            
            # 相关性分析
            f.write("7. CORRELATION ANALYSIS\n")
            f.write("-" * 24 + "\n")
            correlation = self.df['PES_score'].corr(self.df['sequence_length'])
            f.write(f"PES Score vs Sequence Length correlation: {correlation:.4f}\n")
            if abs(correlation) < 0.1:
                f.write("Interpretation: Very weak correlation\n")
            elif abs(correlation) < 0.3:
                f.write("Interpretation: Weak correlation\n")
            elif abs(correlation) < 0.5:
                f.write("Interpretation: Moderate correlation\n")
            else:
                f.write("Interpretation: Strong correlation\n")
            f.write("\n")
            
            # 顶级生物标志物
            if len(biomarkers) > 0:
                f.write("8. TOP 10 POTENTIAL BIOMARKERS\n")
                f.write("-" * 32 + "\n")
                top_biomarkers = biomarkers.nlargest(10, 'PES_score')
                for idx, (_, row) in enumerate(top_biomarkers.iterrows(), 1):
                    protein_id = row['protein_id'][:50] + '...' if len(row['protein_id']) > 50 else row['protein_id']
                    f.write(f"{idx:2d}. {protein_id}\n")
                    f.write(f"    PES Score: {row['PES_score']:.4f}, Length: {row['sequence_length']}\n")
                f.write("\n")
            
            f.write("Report generated successfully!\n")
            f.write("For detailed visualizations, please check the PNG files in this directory.\n")
        
        print("✓ Text report saved")

def main():
    """主函数"""
    csv_file = "../neutrophil_mane_proteins_predictions_ensemble.csv"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        print("Please make sure the CSV file is in the current directory.")
        return
    
    # 创建分析器
    analyzer = ProteinVisualizationAnalyzer(csv_file)
    
    # 生成完整报告
    output_dir = analyzer.generate_report()
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Visualizations and report saved to: {output_dir}/")
    print(f"📈 Files generated:")
    print(f"   - 01_overview_analysis.png")
    print(f"   - 02_detailed_analysis.png") 
    print(f"   - 03_biomarker_analysis.png")
    print(f"   - 04_statistical_summary.png")
    print(f"   - protein_analysis_report.txt")

if __name__ == "__main__":
    main()