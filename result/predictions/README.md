# 预测结果文件

本目录包含中性粒细胞相关蛋白质的预测结果。

## 文件说明

- `neutrophil_human_predictions.csv` - 人类层面模型预测结果
- `neutrophil_immune_ensemble_predictions.csv` - 免疫细胞层面集成预测结果
- `prediction_comparison_results.csv` - 完整的对比分析数据
- `disagreement_proteins.csv` - 预测不一致的蛋白质数据

## 数据格式

每个CSV文件包含以下主要字段：
- `protein_id` - 蛋白质标识符
- `PES_score` - 蛋白质必需性分数 (0-1)
- `prediction` - 预测类别 (Essential/Non-essential)
- `confidence` - 预测置信度
