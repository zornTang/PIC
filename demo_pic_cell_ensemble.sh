#!/bin/bash

# ==============================================================================
# PIC-cell 集成学习运行脚本
# 功能：训练细胞系特异性PIC模型并构建集成模型
# 作者：PIC项目组
# ==============================================================================

set -euo pipefail  # 严格错误处理

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 显示标题
show_banner() {
    echo "=================================================================="
    echo "                    PIC-cell 集成学习系统"
    echo "               Cell-specific PIC Model Ensemble"
    echo "=================================================================="
    echo ""
}

# 检查系统依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 检查conda
    if ! command -v conda &> /dev/null; then
        log_error "Conda未安装或不在PATH中"
        log_error "请安装Anaconda或Miniconda"
        exit 1
    fi
    
    log_success "系统依赖检查通过"
}

# 初始化conda环境
setup_conda_env() {
    log_info "初始化conda环境..."
    
    # 初始化conda
    if ! eval "$(conda shell.bash hook)" 2>/dev/null; then
        # 尝试常见的conda安装路径
        local conda_paths=(
            "$HOME/miniconda3/etc/profile.d/conda.sh"
            "$HOME/anaconda3/etc/profile.d/conda.sh"
            "/opt/conda/etc/profile.d/conda.sh"
            "/usr/local/miniconda3/etc/profile.d/conda.sh"
        )
        
        local conda_found=false
        for path in "${conda_paths[@]}"; do
            if [[ -f "$path" ]]; then
                source "$path"
                conda_found=true
                break
            fi
        done
        
        if [[ "$conda_found" == false ]]; then
            log_error "无法初始化conda环境"
            exit 1
        fi
    fi
    
    # 检查PIC环境
    if ! conda env list | grep -q "^PIC\s"; then
        log_error "PIC环境不存在"
        echo ""
        echo "请先创建PIC环境："
        echo "  conda create -n PIC python=3.8 -y"
        echo "  conda activate PIC"
        echo "  pip install torch pandas scikit-learn numpy transformers"
        exit 1
    fi
    
    # 激活PIC环境
    log_info "激活PIC环境..."
    conda activate PIC
    
    # 验证环境
    if [[ "$CONDA_DEFAULT_ENV" != "PIC" ]]; then
        log_error "PIC环境激活失败"
        exit 1
    fi
    
    log_success "PIC环境激活成功"
}

# 检查必要文件
check_required_files() {
    log_info "检查必要文件..."
    
    local required_files=(
        "data/cell_data.pkl"
        "data/cell_line_meta_info.csv"
        "pretrained_model/esm2_t33_650M_UR50D.pt"
        "train_and_ensemble_all_cells.py"
    )
    
    local missing_files=()
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "缺少必要文件："
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    log_success "所有必要文件存在"
}

# 检查Python包
check_python_packages() {
    log_info "检查Python包..."
    
    local packages=("pandas" "torch" "sklearn" "numpy" "transformers")
    local missing_packages=()
    
    for package in "${packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "缺少Python包："
        for package in "${missing_packages[@]}"; do
            echo "  - $package"
        done
        echo ""
        echo "请安装缺少的包："
        echo "  pip install ${missing_packages[*]}"
        exit 1
    fi
    
    log_success "Python包检查通过"
}

# 创建输出目录
create_output_dirs() {
    log_info "创建输出目录..."
    
    local dirs=(
        "result/model_train_results"
        "result/seq_embedding"
        "result/ensemble_results"
        "logs"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "输出目录创建完成"
}

# 显示运行模式选择
show_menu() {
    echo ""
    echo "请选择运行模式："
    echo "=================================================================="
    echo "1. 快速测试    - 训练5个细胞系 (约30分钟)"
    echo "2. 免疫细胞系  - 训练14个免疫细胞系 (约2-4小时)"
    echo "3. 完整训练    - 训练所有323个细胞系 (约50-150小时)"
    echo "4. 仅集成      - 使用已训练模型进行集成"
    echo "5. 策略比较    - 比较不同投票策略"
    echo "=================================================================="
    echo ""
}

# 执行Python脚本
run_python_script() {
    local script_args=("$@")
    local log_file="logs/pic_cell_$(date +%Y%m%d_%H%M%S).log"
    
    log_info "开始执行Python脚本..."
    log_info "日志文件: $log_file"
    
    echo "执行命令: python ${script_args[*]}"
    
    # 执行脚本并记录日志
    if python "${script_args[@]}" 2>&1 | tee "$log_file"; then
        log_success "Python脚本执行成功"
        return 0
    else
        log_error "Python脚本执行失败"
        log_error "详细日志请查看: $log_file"
        exit 1
    fi
}

# 显示结果
show_results() {
    log_info "显示运行结果..."
    
    if [[ -d "result/ensemble_results" ]]; then
        echo ""
        echo "集成结果文件："
        ls -la result/ensemble_results/
        echo ""
        
        # 显示性能指标
        if [[ -f "result/ensemble_results/metrics_soft.csv" ]]; then
            echo "软投票策略性能指标："
            python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('result/ensemble_results/metrics_soft.csv')
    print('  Accuracy:  {:.4f}'.format(df['accuracy'].iloc[0]))
    print('  Precision: {:.4f}'.format(df['precision'].iloc[0]))
    print('  Recall:    {:.4f}'.format(df['recall'].iloc[0]))
    print('  F1-Score:  {:.4f}'.format(df['f1_score'].iloc[0]))
    print('  AUC-ROC:   {:.4f}'.format(df['auc_roc'].iloc[0]))
    print('  AUC-PR:    {:.4f}'.format(df['auc_pr'].iloc[0]))
except Exception as e:
    print('无法读取性能指标:', e)
    sys.exit(1)
"
        fi
    fi
    
    echo ""
    echo "=================================================================="
    echo "输出位置："
    echo "  模型文件: result/model_train_results/"
    echo "  集成结果: result/ensemble_results/"
    echo "  日志文件: logs/"
    echo "=================================================================="
}

# 主函数
main() {
    show_banner
    
    # 环境检查
    check_dependencies
    setup_conda_env
    check_required_files
    check_python_packages
    create_output_dirs
    
    # 检查是否为后台运行模式
    local choice
    if [[ -t 0 ]]; then
        # 交互式模式 - 标准输入是终端
        show_menu
        read -p "请输入选择 (1-5): " choice
    else
        # 后台运行模式 - 使用默认选择
        choice="${1:-2}"  # 默认选择免疫细胞系模式
        log_info "后台运行模式，使用默认选择: $choice"
        case $choice in
            1) log_info "默认模式: 快速测试" ;;
            2) log_info "默认模式: 免疫细胞系" ;;
            3) log_info "默认模式: 完整训练" ;;
            4) log_info "默认模式: 仅集成" ;;
            5) log_info "默认模式: 策略比较" ;;
            *) 
                log_error "无效的默认选择: $choice，使用免疫细胞系模式"
                choice=2
                ;;
        esac
    fi
    
    # 设置环境变量
    export MKL_SERVICE_FORCE_INTEL=1
    
    # 检测和配置GPU设备
    local gpu_device="cpu"
    local target_gpu_id=7  # 使用GPU 7
    
    log_info "检查GPU设备状态..."
    
    # 检查nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_info "nvidia-smi 可用，检查GPU状态..."
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
            echo "  $line"
        done
    else
        log_warning "nvidia-smi 不可用"
    fi
    
    # 检查PyTorch CUDA支持
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        log_info "PyTorch检测到 $gpu_count 个GPU设备"
        
        # 显示PyTorch可见的GPU
        python -c "
import torch
print('PyTorch CUDA版本:', torch.version.cuda)
print('PyTorch可见的GPU:')
for i in range(torch.cuda.device_count()):
    print('  GPU ' + str(i) + ': ' + torch.cuda.get_device_name(i))
    props = torch.cuda.get_device_properties(i)
    print('    内存: {:.1f} GB'.format(props.total_memory / 1024**3))
"
        
        # 检查目标GPU是否可用
        if [[ $gpu_count -gt $target_gpu_id ]]; then
            gpu_device="cuda:$target_gpu_id"
            log_info "使用 GPU $target_gpu_id"
            
            # 测试GPU是否真的可用
            if python -c "
import torch
try:
    device = torch.device('cuda:$target_gpu_id')
    torch.cuda.set_device(device)
    x = torch.tensor([1.0]).to(device)
    print('GPU $target_gpu_id 测试成功')
    exit(0)
except Exception as e:
    print('GPU $target_gpu_id 测试失败:', e)
    exit(1)
" 2>/dev/null; then
                log_success "GPU $target_gpu_id 可用且测试通过"
                # 不设置CUDA_VISIBLE_DEVICES，让PyTorch直接访问指定GPU
            else
                log_error "GPU $target_gpu_id 测试失败，可能被其他进程占用"
                log_info "尝试使用 GPU 0"
                gpu_device="cuda:0"
                target_gpu_id=0
            fi
        else
            log_warning "PyTorch检测不到GPU $target_gpu_id，可用GPU数量: $gpu_count"
            if [[ $gpu_count -gt 0 ]]; then
                gpu_device="cuda:0"
                log_info "改用 GPU 0"
            fi
        fi
    else
        log_warning "PyTorch CUDA不可用，使用CPU"
    fi
    
    log_info "最终使用设备: $gpu_device"
    
    # 基础参数
    local base_args=(
        "train_and_ensemble_all_cells.py"
        "--data_path" "data/cell_data.pkl"
        "--esm_model_path" "pretrained_model/esm2_t33_650M_UR50D.pt"
        "--device" "$gpu_device"
        "--voting_strategy" "soft"
    )
    
    # 根据选择执行不同的任务
    case $choice in
        1)
            log_info "开始快速测试模式..."
            run_python_script "${base_args[@]}" \
                --cell_line_meta_file "data/cell_line_meta_info.csv" \
                --num_epochs 10 \
                --test_size 500 \
                --specific_cell_lines "A549,MCF7,HT-29,HeLa,PC-3" \
                --overwrite
            ;;
        2)
            log_info "开始免疫细胞系模式..."
            run_python_script "${base_args[@]}" \
                --cell_line_meta_file "data/cell_line_meta_info.csv" \
                --num_epochs 15 \
                --test_size 1000 \
                --specific_cell_lines "ARH-77,IM-9,KMS-11,L-363,LP-1,OCI-AML2,OCI-AML3,OCI-LY-19,OPM-2,ROS-50,RPMI-8226,SU-DHL-10,SU-DHL-5,SU-DHL-8" \
                --overwrite
            ;;
        3)
            log_warning "完整训练模式预计需要50-150小时"
            local confirm
            if [[ -t 0 ]]; then
                # 交互式模式
                read -p "确认继续? (y/N): " confirm
            else
                # 后台运行模式 - 自动确认
                confirm="y"
                log_info "后台运行模式，自动确认完整训练"
            fi
            
            if [[ $confirm =~ ^[yY]$ ]]; then
                log_info "开始完整训练模式..."
                run_python_script "${base_args[@]}" \
                    --cell_line_meta_file "data/cell_line_meta_info.csv" \
                    --num_epochs 15 \
                    --test_size 1000
            else
                log_info "操作已取消"
                exit 0
            fi
            ;;
        4)
            log_info "开始仅集成模式..."
            run_python_script "${base_args[@]}" \
                --test_size 1000 \
                --skip_training
            ;;
        5)
            log_info "开始策略比较模式..."
            run_python_script "${base_args[@]}" \
                --voting_strategy "all" \
                --test_size 1000 \
                --skip_training
            ;;
        *)
            log_error "无效选择: $choice"
            exit 1
            ;;
    esac
    
    # 显示结果
    show_results
    log_success "PIC-cell 集成学习完成！"
}

# 错误处理
trap 'log_error "脚本执行过程中发生错误，退出码: $?"' ERR

# 检查是否需要后台运行
if [[ "${1:-}" == "--background" || "${1:-}" == "-bg" ]]; then
    log_info "启动后台运行模式..."
    log_file="pic_cell_background_$(date +%Y%m%d_%H%M%S).log"
    
    # 移除--background参数
    shift
    
    # 获取运行模式参数
    run_mode="${1:-2}"  # 默认免疫细胞系模式
    
    echo "=================================================================="
    echo "后台运行配置:"
    case $run_mode in
        1) echo "  运行模式: 快速测试 (5个细胞系)" ;;
        2) echo "  运行模式: 免疫细胞系 (14个细胞系)" ;;
        3) echo "  运行模式: 完整训练 (323个细胞系)" ;;
        4) echo "  运行模式: 仅集成" ;;
        5) echo "  运行模式: 策略比较" ;;
        *) echo "  运行模式: 默认 (免疫细胞系)"; run_mode=2 ;;
    esac
    echo "  日志文件: $log_file"
    echo "=================================================================="
    
    # 在后台运行，忽略SIGHUP信号
    nohup bash "$0" "$run_mode" > "$log_file" 2>&1 &
    
    echo "脚本已在后台启动"
    echo "进程ID: $!"
    echo "=================================================================="
    echo "监控进度: tail -f $log_file"
    echo "停止进程: kill $!"
    echo "=================================================================="
    
    exit 0
fi

# 运行主函数
main "$@" 