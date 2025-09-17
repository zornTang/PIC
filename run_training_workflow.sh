#!/bin/bash
# PIC项目专门的训练阶段工作流程脚本
# 包含序列嵌入提取和模型训练的完整流程

set -e  # 遇到错误立即退出

echo "🚀 开始执行PIC项目训练阶段工作流程..."
echo "📅 开始时间: $(date)"

# ============================================================================
# 配置参数
# ============================================================================

# 默认参数
DEFAULT_DEVICE="cuda:7"
DEFAULT_BATCH_SIZE=64
DEFAULT_NUM_EPOCHS=15
DEFAULT_LEARNING_RATE=1e-5
DEFAULT_OVERWRITE=false

# 解析命令行参数
DEVICE=${1:-$DEFAULT_DEVICE}
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
NUM_EPOCHS=${3:-$DEFAULT_NUM_EPOCHS}
LEARNING_RATE=${4:-$DEFAULT_LEARNING_RATE}
OVERWRITE=${5:-$DEFAULT_OVERWRITE}

echo "⚙️  训练配置:"
echo "   - GPU设备: $DEVICE"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - 训练轮数: $NUM_EPOCHS"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 覆盖已有模型: $OVERWRITE"

# ============================================================================
# 环境检查
# ============================================================================

echo ""
echo "📋 检查环境和依赖..."

# 检查虚拟环境
if ! conda info --envs | grep -q "PIC"; then
    echo "❌ 错误: 未找到PIC虚拟环境"
    echo "请先创建并激活PIC环境: conda activate PIC"
    exit 1
fi

# 激活环境
echo "🔧 激活PIC环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate PIC

# 检查GPU可用性
echo "🖥️  检查GPU可用性..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ 错误: 无法访问GPU"
    exit 1
fi

# 显示GPU信息
echo "✅ GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
    echo "   GPU $line"
done

# ============================================================================
# 文件检查
# ============================================================================

echo ""
echo "📁 检查必要文件..."

# 必需的数据文件
required_files=(
    "data/human_data.pkl"
    "data/cell_data.pkl"
    "data/cell_line_meta_info.csv"
    "pretrained_model/esm2_t33_650M_UR50D.pt"
)

# 必需的代码文件
required_code_files=(
    "code/embedding.py"
    "code/main.py"
    "code/train_all_cell_lines.py"
    "code/module/PIC.py"
    "code/module/earlystopping.py"
)

# 检查数据文件
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 未找到必要文件 $file"
        exit 1
    else
        echo "✅ 找到: $file"
    fi
done

# 检查代码文件
for file in "${required_code_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 未找到必要代码文件 $file"
        exit 1
    else
        echo "✅ 找到: $file"
    fi
done

# ============================================================================
# 创建输出目录
# ============================================================================

echo ""
echo "📂 创建输出目录..."

# 创建必要的输出目录
output_dirs=(
    "result"
    "result/seq_embedding"
    "result/model_train_results"
    "logs"
)

for dir in "${output_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ 创建目录: $dir"
    else
        echo "✅ 目录已存在: $dir"
    fi
done

# ============================================================================
# 阶段1: 序列嵌入提取
# ============================================================================

echo ""
echo "🧬 阶段1: 序列嵌入提取"
echo "=" | tr '=' '=' | head -c 60; echo

# 检查是否已有嵌入文件
embedding_count=$(find result/seq_embedding -name "*.pt" 2>/dev/null | wc -l)
echo "📊 当前嵌入文件数量: $embedding_count"

if [ "$embedding_count" -gt 0 ] && [ "$OVERWRITE" != "true" ]; then
    echo "✅ 发现已有序列嵌入文件，跳过嵌入提取阶段"
    echo "   如需重新提取，请使用参数: $0 $DEVICE $BATCH_SIZE $NUM_EPOCHS $LEARNING_RATE true"
else
    echo "🔄 开始提取序列嵌入..."
    
    # 记录开始时间
    embedding_start_time=$(date +%s)
    
    # 执行嵌入提取
    python code/embedding.py \
        --data_path data/human_data.pkl \
        --fasta_file data/human_sequences.fa \
        --model_name pretrained_model/esm2_t33_650M_UR50D.pt \
        --label_name human \
        --output_dir result/seq_embedding \
        --device $DEVICE \
        --toks_per_batch $((BATCH_SIZE * 150))
    
    # 记录结束时间
    embedding_end_time=$(date +%s)
    embedding_duration=$((embedding_end_time - embedding_start_time))
    
    # 检查嵌入提取结果
    new_embedding_count=$(find result/seq_embedding -name "*.pt" 2>/dev/null | wc -l)
    echo "✅ 序列嵌入提取完成"
    echo "   - 耗时: ${embedding_duration}秒"
    echo "   - 生成文件数: $new_embedding_count"
    
    if [ "$new_embedding_count" -eq 0 ]; then
        echo "❌ 错误: 序列嵌入提取失败，未生成任何文件"
        exit 1
    fi
fi

# ============================================================================
# 阶段2: 人类层面模型训练
# ============================================================================

echo ""
echo "👤 阶段2: 人类层面模型训练"
echo "=" | tr '=' '=' | head -c 60; echo

# 检查人类模型是否已存在
human_model_path="result/model_train_results/PIC_human/PIC_human_model.pth"

if [ -f "$human_model_path" ] && [ "$OVERWRITE" != "true" ]; then
    echo "✅ 人类层面模型已存在，跳过训练: $human_model_path"
else
    echo "🔄 开始训练人类层面模型..."
    
    # 记录开始时间
    human_start_time=$(date +%s)
    
    # 执行人类层面模型训练
    python code/main.py \
        --data_path data/human_data.pkl \
        --feature_dir result/seq_embedding \
        --label_name human \
        --save_path result/model_train_results \
        --device $DEVICE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --test_ratio 0.2 \
        --val_ratio 0.1 \
        --linear_drop 0.1 \
        --attn_drop 0.3 \
        --max_length 1000 \
        --feature_length 1280 \
        --input_size 1280 \
        --hidden_size 320 \
        --output_size 1 \
        --random_seed 42
    
    # 记录结束时间
    human_end_time=$(date +%s)
    human_duration=$((human_end_time - human_start_time))
    
    # 检查训练结果
    if [ -f "$human_model_path" ]; then
        echo "✅ 人类层面模型训练完成"
        echo "   - 耗时: ${human_duration}秒"
        echo "   - 模型保存: $human_model_path"
    else
        echo "❌ 错误: 人类层面模型训练失败"
        exit 1
    fi
fi

# ============================================================================
# 阶段3: 免疫细胞系模型训练
# ============================================================================

echo ""
echo "🛡️  阶段3: 免疫细胞系模型训练"
echo "=" | tr '=' '=' | head -c 60; echo

# 目标细胞系列表
target_cell_lines="ARH-77,IM-9,KMS-11,L-363,LP-1,OCI-AML2,OCI-AML3,OCI-LY-19,OPM-2,ROS-50,RPMI-8226,SU-DHL-10,SU-DHL-5,SU-DHL-8"

# 检查已有的免疫细胞系模型
echo "📊 检查已有免疫细胞系模型..."
existing_models=0
IFS=',' read -ra CELL_LINES <<< "$target_cell_lines"
for cell_line in "${CELL_LINES[@]}"; do
    model_path="result/model_train_results/PIC_${cell_line}/PIC_${cell_line}_model.pth"
    if [ -f "$model_path" ]; then
        existing_models=$((existing_models + 1))
    fi
done

total_cell_lines=${#CELL_LINES[@]}
echo "   - 目标细胞系数量: $total_cell_lines"
echo "   - 已有模型数量: $existing_models"

if [ "$existing_models" -eq "$total_cell_lines" ] && [ "$OVERWRITE" != "true" ]; then
    echo "✅ 所有免疫细胞系模型已存在，跳过训练"
else
    echo "🔄 开始训练免疫细胞系模型..."
    
    # 记录开始时间
    immune_start_time=$(date +%s)
    
    # 构建训练参数
    overwrite_flag=""
    if [ "$OVERWRITE" = "true" ]; then
        overwrite_flag="--overwrite"
    fi
    
    # 执行免疫细胞系模型训练
    python code/train_all_cell_lines.py \
        --data_path data/cell_data.pkl \
        --cell_line_meta_file data/cell_line_meta_info.csv \
        --esm_model_path pretrained_model/esm2_t33_650M_UR50D.pt \
        --output_dir result/model_train_results \
        --embedding_dir result/seq_embedding \
        --specific_cell_lines "$target_cell_lines" \
        --device $DEVICE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --test_ratio 0.2 \
        --val_ratio 0.1 \
        --linear_drop 0.1 \
        --attn_drop 0.3 \
        --max_length 1000 \
        --feature_length 1280 \
        --input_size 1280 \
        --hidden_size 320 \
        --output_size 1 \
        --random_seed 42 \
        $overwrite_flag
    
    # 记录结束时间
    immune_end_time=$(date +%s)
    immune_duration=$((immune_end_time - immune_start_time))
    
    # 检查训练结果
    final_models=0
    for cell_line in "${CELL_LINES[@]}"; do
        model_path="result/model_train_results/PIC_${cell_line}/PIC_${cell_line}_model.pth"
        if [ -f "$model_path" ]; then
            final_models=$((final_models + 1))
        fi
    done
    
    echo "✅ 免疫细胞系模型训练完成"
    echo "   - 耗时: ${immune_duration}秒"
    echo "   - 成功训练模型数: $final_models/$total_cell_lines"
    
    if [ "$final_models" -eq 0 ]; then
        echo "❌ 错误: 免疫细胞系模型训练失败"
        exit 1
    fi
fi

# ============================================================================
# 训练完成总结
# ============================================================================

echo ""
echo "🎉 训练阶段工作流程完成!"
echo "=" | tr '=' '=' | head -c 60; echo

# 计算总耗时
total_end_time=$(date +%s)
total_start_time=$(date -d "$(head -1 /tmp/training_start_time 2>/dev/null || echo '1 second ago')" +%s 2>/dev/null || echo $total_end_time)
total_duration=$((total_end_time - total_start_time))

echo "📊 训练总结:"
echo "   - 完成时间: $(date)"
echo "   - 总耗时: ${total_duration}秒 ($(($total_duration / 60))分钟)"

# 检查最终结果
echo ""
echo "📁 检查训练结果..."

# 检查序列嵌入
embedding_files=$(find result/seq_embedding -name "*.pt" 2>/dev/null | wc -l)
echo "✅ 序列嵌入文件: $embedding_files 个"

# 检查人类模型
if [ -f "result/model_train_results/PIC_human/PIC_human_model.pth" ]; then
    echo "✅ 人类层面模型: 已完成"
    human_model_size=$(du -h "result/model_train_results/PIC_human/PIC_human_model.pth" | cut -f1)
    echo "   - 模型大小: $human_model_size"
else
    echo "❌ 人类层面模型: 缺失"
fi

# 检查免疫细胞系模型
immune_model_count=0
IFS=',' read -ra CELL_LINES <<< "$target_cell_lines"
for cell_line in "${CELL_LINES[@]}"; do
    model_path="result/model_train_results/PIC_${cell_line}/PIC_${cell_line}_model.pth"
    if [ -f "$model_path" ]; then
        immune_model_count=$((immune_model_count + 1))
    fi
done

echo "✅ 免疫细胞系模型: $immune_model_count/${#CELL_LINES[@]} 个"

# 计算总模型数
total_models=$((immune_model_count + ([ -f "result/model_train_results/PIC_human/PIC_human_model.pth" ] && echo 1 || echo 0)))
echo "✅ 总模型数量: $total_models 个"

# ============================================================================
# 下一步建议
# ============================================================================

echo ""
echo "🚀 下一步操作建议:"
echo "=" | tr '=' '=' | head -c 60; echo

echo "1. 📊 检查项目状态:"
echo "   python check_project_status.py"
echo ""

echo "2. 🔮 执行预测分析:"
echo "   bash run_complete_workflow.sh"
echo ""

echo "3. 🔍 查看训练日志:"
echo "   ls -la logs/"
echo ""

echo "4. 📈 查看模型详情:"
echo "   ls -la result/model_train_results/"
echo ""

echo "5. 🧪 测试单个模型预测:"
echo "   python code/predict.py --model_path result/model_train_results/PIC_human/PIC_human_model.pth --input_fasta neutrophil_mane_proteins.fa --output_file test_predictions.csv --device $DEVICE"

# ============================================================================
# 清理和日志
# ============================================================================

echo ""
echo "🧹 清理临时文件..."

# 清理可能的临时文件
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.log" -type f -size +100M -delete 2>/dev/null || true

# 保存训练配置到日志
training_log="logs/training_workflow_$(date +%Y%m%d_%H%M%S).log"
cat > "$training_log" << EOF
PIC项目训练工作流程日志
========================

执行时间: $(date)
配置参数:
- GPU设备: $DEVICE
- 批次大小: $BATCH_SIZE
- 训练轮数: $NUM_EPOCHS
- 学习率: $LEARNING_RATE
- 覆盖模式: $OVERWRITE

训练结果:
- 序列嵌入文件: $embedding_files 个
- 人类层面模型: $([ -f "result/model_train_results/PIC_human/PIC_human_model.pth" ] && echo "完成" || echo "失败")
- 免疫细胞系模型: $immune_model_count/${#CELL_LINES[@]} 个
- 总模型数量: $total_models 个
- 总耗时: ${total_duration}秒

目标细胞系: $target_cell_lines
EOF

echo "✅ 训练日志已保存: $training_log"

echo ""
echo "🎯 训练阶段工作流程全部完成!"
echo "📁 所有结果保存在 result/ 目录下"
echo "📋 详细日志: $training_log"
echo ""
