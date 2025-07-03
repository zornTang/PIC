#!/bin/bash

# PIC项目清理脚本
# 用于清理不必要的临时文件和缓存

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=================================================================="
echo "                    PIC项目清理脚本"
echo "=================================================================="
echo ""

# 1. 清理Python缓存文件
log_info "清理Python缓存文件..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -name "*.pyo" -type f -delete 2>/dev/null || true
log_success "Python缓存文件清理完成"

# 2. 清理过期日志文件（保留最近3个）
log_info "清理过期日志文件..."
if [[ -d "logs" ]]; then
    cd logs
    # 保留最新的3个日志文件
    ls -t pic_cell_*.log 2>/dev/null | tail -n +4 | xargs -r rm -f
    cd ..
    log_success "过期日志文件清理完成"
fi

# 3. 清理训练日志文件（保留最近3个）
log_info "清理过期训练日志文件..."
if [[ -d "result/model_train_results" ]]; then
    cd result/model_train_results
    ls -t training_log_*.log 2>/dev/null | tail -n +4 | xargs -r rm -f
    cd ../..
    log_success "过期训练日志文件清理完成"
fi

# 4. 清理集成日志文件（保留最近3个）
log_info "清理过期集成日志文件..."
if [[ -d "result/ensemble_results" ]]; then
    cd result/ensemble_results
    ls -t pic_cell_workflow_*.log 2>/dev/null | tail -n +4 | xargs -r rm -f
    ls -t ensemble_log_*.log 2>/dev/null | tail -n +4 | xargs -r rm -f
    cd ../..
    log_success "过期集成日志文件清理完成"
fi

# 5. 清理空的.pt文件
log_info "清理空的序列嵌入文件..."
if [[ -d "result/seq_embedding" ]]; then
    empty_files=$(find result/seq_embedding/ -name "*.pt" -size 0 2>/dev/null | wc -l)
    if [[ $empty_files -gt 0 ]]; then
        find result/seq_embedding/ -name "*.pt" -size 0 -delete
        log_success "清理了 $empty_files 个空的.pt文件"
    else
        log_info "没有发现空的.pt文件"
    fi
fi

# 6. 显示清理后的统计信息
echo ""
log_info "清理完成！当前项目大小分布："
du -sh * 2>/dev/null | sort -hr | head -10

echo ""
echo "=================================================================="
echo "清理完成！"
echo "如需清理序列嵌入文件(result/seq_embedding/)以节省空间，"
echo "请确认这些文件可以重新生成后手动删除。"
echo "==================================================================" 