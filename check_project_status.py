#!/usr/bin/env python3
"""
PIC项目状态检查脚本
检查项目完整性和结果文件状态
"""

import os
import glob
from pathlib import Path

def check_directory_structure():
    """检查目录结构完整性"""
    print("🏗️  检查项目目录结构...")
    
    required_dirs = [
        'result/seq_embedding',
        'result/model_train_results',
        'result/predictions',
        'result/neutrophil_analysis/visualizations',
        'result/neutrophil_analysis/reports',
        'result/neutrophil_analysis/data',
        'data',
        'code',
        'pretrained_model'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ❌ {directory} - 缺失")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0

def check_model_files():
    """检查模型文件"""
    print("\n🤖 检查训练模型...")
    
    model_dir = 'result/model_train_results'
    if not os.path.exists(model_dir):
        print("  ❌ 模型目录不存在")
        return False
    
    # 检查人类层面模型
    human_model = f"{model_dir}/PIC_human/PIC_human_model.pth"
    if os.path.exists(human_model):
        print("  ✓ 人类层面模型")
    else:
        print("  ❌ 人类层面模型缺失")
    
    # 检查免疫细胞系模型
    cell_lines = ['ARH-77', 'IM-9', 'KMS-11', 'L-363', 'LP-1', 'OCI-AML2', 
                  'OCI-AML3', 'OCI-LY-19', 'OPM-2', 'ROS-50', 'RPMI-8226', 
                  'SU-DHL-10', 'SU-DHL-5', 'SU-DHL-8']
    
    immune_models = 0
    for cell_line in cell_lines:
        model_path = f"{model_dir}/PIC_{cell_line}/PIC_{cell_line}_model.pth"
        if os.path.exists(model_path):
            immune_models += 1
    
    print(f"  ✓ 免疫细胞系模型: {immune_models}/{len(cell_lines)}")
    
    return immune_models >= 10  # 至少需要10个模型

def check_embedding_files():
    """检查嵌入文件"""
    print("\n🧬 检查序列嵌入文件...")
    
    embedding_dir = 'result/seq_embedding'
    if not os.path.exists(embedding_dir):
        print("  ❌ 嵌入目录不存在")
        return False
    
    embedding_files = glob.glob(f"{embedding_dir}/*.pt")
    print(f"  ✓ 嵌入文件数量: {len(embedding_files)}")
    
    return len(embedding_files) > 1000  # 应该有大量嵌入文件

def check_prediction_files():
    """检查预测结果文件"""
    print("\n🔮 检查预测结果...")
    
    prediction_files = [
        'result/predictions/neutrophil_human_predictions.csv',
        'result/predictions/neutrophil_immune_ensemble_predictions.csv'
    ]
    
    all_exist = True
    for file_path in prediction_files:
        if os.path.exists(file_path):
            # 检查文件大小
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {os.path.basename(file_path)} ({size:.1f} KB)")
        else:
            print(f"  ❌ {os.path.basename(file_path)} - 缺失")
            all_exist = False
    
    return all_exist

def check_analysis_results():
    """检查分析结果"""
    print("\n📊 检查分析结果...")
    
    # 检查可视化文件
    viz_files = [
        'result/neutrophil_analysis/visualizations/01_overview_comparison.png',
        'result/neutrophil_analysis/visualizations/02_detailed_analysis.png',
        'result/neutrophil_analysis/visualizations/03_biomarker_analysis.png',
        'result/neutrophil_analysis/visualizations/04_functional_analysis.png'
    ]
    
    viz_count = 0
    for file_path in viz_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {os.path.basename(file_path)} ({size:.1f} KB)")
            viz_count += 1
        else:
            print(f"  ❌ {os.path.basename(file_path)} - 缺失")
    
    # 检查报告文件
    report_files = [
        'result/neutrophil_analysis/reports/enhanced_comparison_report.md',
        'result/neutrophil_analysis/reports/enhanced_comparison_report.txt'
    ]
    
    report_count = 0
    for file_path in report_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {os.path.basename(file_path)} ({size:.1f} KB)")
            report_count += 1
        else:
            print(f"  ❌ {os.path.basename(file_path)} - 缺失")
    
    # 检查数据文件
    data_files = [
        'result/neutrophil_analysis/data/prediction_comparison_results.csv',
        'result/neutrophil_analysis/data/disagreement_proteins.csv'
    ]
    
    data_count = 0
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {os.path.basename(file_path)} ({size:.1f} KB)")
            data_count += 1
        else:
            print(f"  ❌ {os.path.basename(file_path)} - 缺失")
    
    return viz_count == 4 and report_count == 2 and data_count == 2

def check_documentation():
    """检查文档文件"""
    print("\n📚 检查项目文档...")

    doc_files = [
        'README.md',
        'PIC_PROJECT_DOCUMENTATION.md',
        'run_complete_workflow.sh'
    ]

    doc_count = 0
    for file_path in doc_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {file_path} ({size:.1f} KB)")
            doc_count += 1
        else:
            print(f"  ❌ {file_path} - 缺失")

    return doc_count >= 2

def generate_status_report():
    """生成状态报告"""
    print("\n" + "=" * 60)
    print("PIC项目状态检查报告")
    print("=" * 60)
    
    checks = [
        ("目录结构", check_directory_structure()),
        ("训练模型", check_model_files()),
        ("序列嵌入", check_embedding_files()),
        ("预测结果", check_prediction_files()),
        ("分析结果", check_analysis_results()),
        ("项目文档", check_documentation())
    ]
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    print(f"\n📋 检查结果: {passed}/{total} 项通过")
    
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check_name}")
    
    if passed == total:
        print("\n🎉 项目状态: 完整")
        print("✨ 所有组件都已正确安装和配置")
        print("🚀 可以执行完整的分析流程")
    else:
        print(f"\n⚠️  项目状态: 不完整 ({total-passed}项缺失)")
        print("🔧 请检查缺失的组件并重新运行相关步骤")
    
    # 统计信息
    print(f"\n📊 项目统计:")
    
    # 嵌入文件数量
    embedding_files = glob.glob('result/seq_embedding/*.pt')
    print(f"  • 序列嵌入文件: {len(embedding_files):,}个")
    
    # 模型文件数量
    model_files = glob.glob('result/model_train_results/*/PIC_*_model.pth')
    print(f"  • 训练模型: {len(model_files)}个")
    
    # 可视化文件数量
    viz_files = glob.glob('result/neutrophil_analysis/visualizations/*.png')
    print(f"  • 可视化图表: {len(viz_files)}个")
    
    # 报告文件数量
    report_files = glob.glob('result/neutrophil_analysis/reports/*')
    print(f"  • 分析报告: {len(report_files)}个")
    
    print("\n" + "=" * 60)

def main():
    """主函数"""
    print("🔍 PIC项目状态检查工具")
    print("检查项目完整性和结果文件状态\n")
    
    generate_status_report()
    
    print("\n💡 提示:")
    print("  • 如需重新运行分析: python compare_predictions.py")
    print("  • 如需查看完整文档: cat PIC_PROJECT_DOCUMENTATION.md")
    print("  • 如需执行完整流程: bash run_complete_workflow.sh")

if __name__ == "__main__":
    main()
