#!/bin/bash
# PIC项目完整工作流程执行脚本

set -e  # 遇到错误立即退出

echo "🚀 开始执行PIC项目完整工作流程..."

# 检查环境
echo "📋 检查环境..."
if ! conda info --envs | grep -q "PIC"; then
    echo "❌ 错误: 未找到PIC虚拟环境"
    exit 1
fi

# 激活环境
echo "🔧 激活PIC环境..."
conda activate PIC

# 检查项目状态
echo "🏗️  检查项目状态..."
python check_project_status.py

# 检查必要文件
echo "📁 检查必要文件..."
required_files=(
    "data/human_data.pkl"
    "data/cell_data.pkl"
    "pretrained_model/esm2_t33_650M_UR50D.pt"
    "neutrophil_mane_proteins.fa"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 未找到必要文件 $file"
        exit 1
    fi
done

echo "✅ 所有必要文件检查完成"

# 检查预测结果是否已存在
if [ -f "result/predictions/neutrophil_human_predictions.csv" ] && [ -f "result/predictions/neutrophil_immune_ensemble_predictions.csv" ]; then
    echo "✅ 预测结果文件已存在，跳过预测步骤"
else
    echo "⚠️  预测结果文件不完整，需要重新预测"

    # 执行预测 (假设模型已训练)
    if [ -f "result/model_train_results/PIC_human/PIC_human_model.pth" ]; then
        echo "🔮 执行人类层面预测..."
        python code/predict.py \
            --model_path result/model_train_results/PIC_human/PIC_human_model.pth \
            --input_fasta neutrophil_mane_proteins.fa \
            --output_file result/predictions/neutrophil_human_predictions.csv \
            --device cuda:7 \
            --batch_size 64
    else
        echo "❌ 错误: 未找到人类层面模型"
        exit 1
    fi

    # 执行免疫层面集成预测
    echo "🛡️  执行免疫层面集成预测..."
    python code/predict.py \
        --ensemble_mode \
        --model_dir result/model_train_results \
        --input_fasta neutrophil_mane_proteins.fa \
        --output_file result/predictions/neutrophil_immune_ensemble_predictions.csv \
        --device cuda:7 \
        --batch_size 64 \
        --voting_strategy weighted_average
fi

# 执行对比分析
echo "📊 执行对比分析..."
python compare_predictions.py

# 最终状态检查
echo "🔍 最终状态检查..."
python check_project_status.py

echo ""
echo "🎉 完整工作流程执行完成!"
echo "📁 结果保存在: result/neutrophil_analysis/"
echo "📊 可视化图表: result/neutrophil_analysis/visualizations/"
echo "📄 分析报告: result/neutrophil_analysis/reports/"
echo "📈 查看详细报告: cat result/neutrophil_analysis/reports/enhanced_comparison_report.md"
