#!/usr/bin/env python3
"""
测试第四张图生成的脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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

def test_pie_chart():
    """测试饼图生成"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Test Functional Analysis Visualization',
                 fontsize=16, fontweight='bold', y=0.98)

    # 测试数据
    test_data = ['Category A', 'Category B', 'Category C', 'Category D']
    test_values = [30, 25, 25, 20]
    colors = [NATURE_COLORS['primary_blue'], NATURE_COLORS['primary_green'],
             NATURE_COLORS['primary_orange'], NATURE_COLORS['primary_purple']]

    # 第一个饼图
    ax1 = axes[0, 0]
    wedges, texts, autotexts = ax1.pie(test_values,
                                      labels=test_data,
                                      autopct='%1.1f%%', colors=colors,
                                      textprops={'fontsize': 9, 'fontfamily': 'sans-serif'})
    
    # 设置标签文字字体
    for text in texts:
        text.set_fontfamily('sans-serif')
        text.set_fontsize(9)
    
    # 设置百分比文字字体
    for autotext in autotexts:
        autotext.set_fontfamily('sans-serif')
        autotext.set_fontsize(8)
        
    ax1.set_title('Test Pie Chart 1\nWith Font Fix',
                 fontsize=12, fontweight='bold')

    # 第二个饼图
    ax2 = axes[0, 1]
    wedges, texts, autotexts = ax2.pie(test_values,
                                      labels=test_data,
                                      autopct='%1.1f%%', colors=colors,
                                      textprops={'fontsize': 9, 'fontfamily': 'sans-serif'})
    
    # 设置标签文字字体
    for text in texts:
        text.set_fontfamily('sans-serif')
        text.set_fontsize(9)
    
    # 设置百分比文字字体
    for autotext in autotexts:
        autotext.set_fontfamily('sans-serif')
        autotext.set_fontsize(8)
        
    ax2.set_title('Test Pie Chart 2\nWith Font Fix',
                 fontsize=12, fontweight='bold')

    # 第三个饼图
    ax3 = axes[0, 2]
    wedges, texts, autotexts = ax3.pie(test_values,
                                      labels=test_data,
                                      autopct='%1.1f%%', colors=colors,
                                      textprops={'fontsize': 9, 'fontfamily': 'sans-serif'})
    
    # 设置标签文字字体
    for text in texts:
        text.set_fontfamily('sans-serif')
        text.set_fontsize(9)
    
    # 设置百分比文字字体
    for autotext in autotexts:
        autotext.set_fontfamily('sans-serif')
        autotext.set_fontsize(8)
        
    ax3.set_title('Test Pie Chart 3\nWith Font Fix',
                 fontsize=12, fontweight='bold')

    # 其他子图留空
    for i in range(3):
        axes[1, i].axis('off')
        axes[1, i].text(0.5, 0.5, f'Test Area {i+4}', 
                       transform=axes[1, i].transAxes,
                       ha='center', va='center',
                       fontsize=14, fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig('result/neutrophil_analysis/visualizations/test_04_functional_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Test chart generated successfully")
    print("✓ Saved as: result/neutrophil_analysis/visualizations/test_04_functional_analysis.png")

if __name__ == "__main__":
    test_pie_chart()
