#!/bin/bash

# PIC-cell 后台进程监控脚本

echo "=================================================================="
echo "              PIC-cell 后台进程监控"
echo "=================================================================="

# 查找PIC-cell相关进程
echo "正在运行的PIC-cell进程:"
ps aux | grep -E "(demo_pic_cell|train_and_ensemble)" | grep -v grep

echo ""
echo "=================================================================="

# 查找日志文件
echo "可用的日志文件:"
ls -la *.log 2>/dev/null || echo "未找到日志文件"
ls -la logs/*.log 2>/dev/null || echo "logs目录下未找到日志文件"

echo ""
echo "=================================================================="

# 提供操作选项
echo "监控选项:"
echo "1. 实时查看最新日志"
echo "2. 查看GPU使用情况"
echo "3. 查看磁盘使用情况"
echo "4. 停止所有PIC-cell进程"
echo "=================================================================="

read -p "请选择操作 (1-4): " choice

case $choice in
    1)
        # 找到最新的日志文件
        latest_log=$(ls -t *.log logs/*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            echo "监控日志文件: $latest_log"
            echo "按 Ctrl+C 退出监控"
            tail -f "$latest_log"
        else
            echo "未找到日志文件"
        fi
        ;;
    2)
        echo "GPU使用情况:"
        nvidia-smi
        ;;
    3)
        echo "磁盘使用情况:"
        df -h .
        echo ""
        echo "result目录大小:"
        du -sh result/ 2>/dev/null || echo "result目录不存在"
        ;;
    4)
        echo "正在停止PIC-cell进程..."
        pkill -f "demo_pic_cell"
        pkill -f "train_and_ensemble"
        echo "进程已停止"
        ;;
    *)
        echo "无效选择"
        ;;
esac 