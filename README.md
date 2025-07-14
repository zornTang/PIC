# PIC (Protein Importance Calculator)

基于ESM2蛋白质语言模型的蛋白质必需性预测系统，实现人类层面和免疫细胞层面的双层预测框架。

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate PIC

# 2. 执行训练阶段（如需要）
bash run_training_workflow.sh

# 3. 检查项目状态
python check_project_status.py

# 4. 运行完整分析
python compare_predictions.py

# 5. 执行完整工作流程
bash run_complete_workflow.sh
```

## 📁 项目结构

```
PIC/
├── code/                   # 核心源代码
├── data/                   # 训练数据
├── result/                 # 所有结果文件
│   ├── seq_embedding/      # 序列嵌入 (65,057个)
│   ├── model_train_results/# 训练模型 (15个)
│   └── neutrophil_analysis/# 分析结果和可视化
└── 🛠️ 工具脚本             # 见下方工具说明
```

## 🛠️ 核心工具

- **`run_training_workflow.sh`** - 专门的训练阶段工作流程脚本
- **`compare_predictions.py`** - 预测结果对比分析 (主要功能)
- **`check_project_status.py`** - 项目状态检查
- **`run_complete_workflow.sh`** - 完整工作流程执行

## 📊 主要发现

- **预测一致性**: 75.9% (948/1,249个蛋白质)
- **人类特异性**: 205个蛋白质 (转录调节因子为主)
- **免疫特异性**: 96个蛋白质 (组蛋白相关为主)
- **共识必需**: 50个蛋白质 (代谢酶为主)

## 📖 详细文档

查看 `PIC_PROJECT_DOCUMENTATION.md` 获取完整的项目文档，包括：
- 详细的工作流程
- 分析结果解释
- 生物学意义
- 使用说明

## 📈 查看结果

```bash
# 查看分析结果
ls result/neutrophil_analysis/

# 阅读分析报告
cat result/neutrophil_analysis/reports/enhanced_comparison_report.md
```

## 📖 详细文档

查看 **`PIC_PROJECT_DOCUMENTATION.md`** 获取：
- 完整的工作流程说明
- 详细的分析结果解释
- 训练参数配置指南
- 故障排除和技术支持

---

*PIC项目 - 蛋白质必需性预测与分析系统 v1.0*
