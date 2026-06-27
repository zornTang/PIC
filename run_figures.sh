#!/usr/bin/env bash
# run_figures.sh — 运行 figure_generation 中整合后的最终出图脚本
# 用法：conda run -n PIC bash run_figures.sh
# 或在可用环境内直接：bash run_figures.sh

set -euo pipefail
cd "$(dirname "$0")"

FIG_DIR="figure_generation/figures_final"
mkdir -p "$FIG_DIR"

LOG="$FIG_DIR/run_figures.log"
: > "$LOG"

ok=0
fail=0

run_script() {
    local label="$1"
    local cmd="$2"
    echo -n "  [$label] ... "
    if eval "$cmd" >> "$LOG" 2>&1; then
        echo "OK"
        (( ok++ )) || true
    else
        echo "FAILED  (see $LOG)"
        (( fail++ )) || true
    fi
}

echo "=========================================="
echo " PIC 论文图表重绘"
echo " 输出目录：$FIG_DIR"
echo "=========================================="

# ── 1. 注意力与第三章特征分析 ───────────────────
echo ""
echo "[第二至三章] 注意力与候选蛋白特征分析图"
run_script "ATP6V1B2注意力位点" "python figure_generation/scripts/make_attention_panel.py"
run_script "密集注意力图"       "python figure_generation/scripts/visualize_attention_dense.py"
run_script "PES模型比较"        "python figure_generation/scripts/make_model_comparison.py"
run_script "综合特征图"         "python figure_generation/scripts/visualize_deep_immune_analysis.py"
run_script "AA组成补图"         "python figure_generation/scripts/redraw_deep_aa_composition_review.py"

# ── 2. 模型性能与消融实验 ───────────────────────
echo ""
echo "[第二章] 模型性能与消融实验图"
run_script "训练与消融汇总" "python figure_generation/scripts/recreate_summary_figures.py"
run_script "CCK8面板"       "python figure_generation/scripts/redraw_cck8_fig10_panels.py"

# ── 3. 第四章图件 ───────────────────────────────
echo ""
echo "[第四章] 对接与表达图"
run_script "对接面板" "python figure_generation/scripts/make_docking_panel.py"
run_script "HPA表达热图" "python figure_generation/scripts/make_hpa_panel.py"
run_script "DepMap面板" "python figure_generation/scripts/make_depmap_panel.py"

echo ""
echo "=========================================="
printf " 完成：%d 成功，%d 失败\n" "$ok" "$fail"
echo " 图片输出于：$FIG_DIR"
echo " 详细日志：  $LOG"
echo "=========================================="
