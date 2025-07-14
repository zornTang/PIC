# PIC项目结果目录

本目录包含PIC项目的所有结果文件，按功能模块组织。

## 📁 目录结构

### 核心结果
- **`seq_embedding/`** - 蛋白质序列嵌入文件 (~65,000个.pt文件)
- **`model_train_results/`** - 训练模型和结果 (15个模型目录)
- **`predictions/`** - 预测结果文件
- **`neutrophil_analysis/`** - 中性粒细胞分析结果

### 详细说明

#### 序列嵌入 (`seq_embedding/`)
- ESM2模型提取的蛋白质序列嵌入
- 每个蛋白质对应一个.pt文件
- 用于模型训练的特征输入

#### 训练模型 (`model_train_results/`)
```
model_train_results/
├── PIC_human/              # 人类层面模型
├── PIC_ARH-77/            # 免疫细胞系模型
├── PIC_IM-9/
└── ... (共15个模型目录)
```

#### 预测结果 (`predictions/`)
- 人类层面预测: `neutrophil_human_predictions.csv`
- 免疫层面预测: `neutrophil_immune_ensemble_predictions.csv`

#### 分析结果 (`neutrophil_analysis/`)
- 可视化图表: `visualizations/`
- 分析报告: `reports/`
- 数据文件: `data/`

## 📖 使用说明

详细的工作流程和使用说明请参考项目根目录的文档：
- **`README.md`** - 项目概览和快速开始
- **`PIC_PROJECT_DOCUMENTATION.md`** - 完整项目文档
