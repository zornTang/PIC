# PIC-cell 集成学习框架指南

## 概述

PIC-cell 是一个基于软投票策略的集成学习框架，通过聚合所有323个细胞系特异性PIC模型的预测结果，构建出高性能的集成模型。

## 核心特性

### 🎯 软投票策略 (Soft Voting Strategy)
- **原理**: 对所有细胞系模型的概率输出进行平均
- **公式**: `P_ensemble = (1/N) * Σ P_i`，其中 N 是模型数量，P_i 是第i个模型的预测概率
- **优势**: 充分利用每个模型的置信度信息，提供更稳定的预测

### 🔬 多层次集成
1. **细胞系特异性模型**: 每个细胞系训练独立的PIC模型
2. **集成层**: 使用软投票策略聚合所有模型预测
3. **后处理**: 计算集成置信度和生物标志物分析

### 📊 评估指标
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- 集成一致性分析

## 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate PIC
export MKL_SERVICE_FORCE_INTEL=1

# 检查环境
python check_environment.py
```

### 2. 使用演示脚本（推荐）

```bash
# 使脚本可执行
chmod +x demo_pic_cell_ensemble.sh

# 运行演示
./demo_pic_cell_ensemble.sh
```

演示脚本提供5种运行模式：
1. **完整流程**: 训练所有323个细胞系 + 集成
2. **测试流程**: 训练5个细胞系 + 集成  
3. **免疫细胞系流程**: 训练14个免疫细胞系 + 集成
4. **只进行集成**: 使用已训练的模型
5. **比较投票策略**: 对比软投票、硬投票等策略

### 3. 直接使用Python脚本

#### 完整训练和集成流程

```bash
python train_and_ensemble_all_cells.py \
    --data_path data/cell_data.pkl \
    --cell_line_meta_file data/cell_line_meta_info.csv \
    --esm_model_path pretrained_model/esm2_t33_650M_UR50D.pt \
    --device cuda:7 \
    --num_epochs 15 \
    --voting_strategy soft \
    --test_size 1000
```

#### 只训练特定细胞系

```bash
python train_and_ensemble_all_cells.py \
    --specific_cell_lines "A549,MCF7,HT-29,HeLa,PC-3" \
    --num_epochs 10 \
    --voting_strategy soft \
    --overwrite
```

#### 只进行集成学习

```bash
python train_and_ensemble_all_cells.py \
    --voting_strategy soft \
    --test_size 1000 \
    --skip_training
```

## 高级用法

### 1. 使用集成框架进行预测

```python
from code.ensemble_pic_cell import PICCellEnsemble

# 初始化集成模型
ensemble = PICCellEnsemble(
    model_dir='result/model_train_results',
    esm_model_path='pretrained_model/esm2_t33_650M_UR50D.pt',
    device='cuda:7'
)

# 加载所有可用模型
ensemble.load_cell_line_models()

# 预测单个蛋白质
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
result = ensemble.predict_single_protein(sequence, voting_strategy='soft')

print(f"集成预测概率: {result['ensemble_probability']:.4f}")
print(f"集成预测结果: {result['ensemble_prediction']}")
print(f"参与模型数量: {result['num_models']}")
```

### 2. 比较不同投票策略

```python
# 评估测试数据
test_data = create_test_dataset('data/cell_data.pkl', sample_size=500)

# 比较所有投票策略
results, comparison_df = ensemble.compare_voting_strategies(test_data)

print("投票策略比较:")
print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']])
```

### 3. 自定义权重投票

```python
# 设置自定义权重（基于模型性能）
custom_weights = {
    'A549': 0.15,
    'MCF7': 0.12,
    'HT-29': 0.10,
    # ... 其他细胞系
}

ensemble.set_model_weights(custom_weights)

# 使用加权投票
result = ensemble.predict_single_protein(sequence, voting_strategy='weighted')
```

## 投票策略详解

### 1. 软投票 (Soft Voting) - 推荐
```
P_ensemble = (1/N) * Σ P_i
```
- **优点**: 利用概率信息，预测更稳定
- **适用**: 大多数场景，特别是模型性能相近时

### 2. 硬投票 (Hard Voting)
```
Label_ensemble = majority_vote(Label_1, Label_2, ..., Label_N)
```
- **优点**: 简单直观，计算快速
- **适用**: 模型预测差异较大时

### 3. 加权投票 (Weighted Voting)
```
P_ensemble = Σ (w_i * P_i) / Σ w_i
```
- **优点**: 考虑模型质量差异
- **适用**: 有明确模型性能排序时

## 输出结果

### 1. 目录结构

```
result/
├── model_train_results/          # 细胞系特异性模型
│   ├── PIC_A549/
│   │   ├── PIC_A549_model.pth
│   │   ├── PIC_A549_val_result.csv
│   │   └── PIC_A549_test_result.csv
│   └── ...
├── ensemble_results/             # 集成学习结果
│   ├── metrics_soft.csv          # 性能指标
│   ├── predictions_soft.csv      # 预测结果
│   ├── voting_strategies_comparison.csv  # 策略比较
│   └── pic_cell_workflow_*.log   # 运行日志
└── seq_embedding/               # 序列嵌入（临时文件）
```

### 2. 性能指标文件 (metrics_soft.csv)

| 指标 | 含义 | 典型范围 |
|------|------|----------|
| accuracy | 准确率 | 0.85-0.95 |
| precision | 精确率 | 0.80-0.92 |
| recall | 召回率 | 0.78-0.90 |
| f1_score | F1分数 | 0.82-0.91 |
| auc_roc | ROC曲线下面积 | 0.88-0.96 |
| auc_pr | PR曲线下面积 | 0.85-0.94 |

### 3. 预测结果文件 (predictions_soft.csv)

| 列名 | 说明 |
|------|------|
| y_true | 真实标签 (0/1) |
| y_pred | 预测标签 (0/1) |
| y_prob | 预测概率 (0-1) |

## 性能优化

### 1. 硬件优化

```bash
# 使用多GPU（如果可用）
--device cuda:7    # 主GPU
--device cuda:1    # 备用GPU

# 调整批处理大小
--batch_size 512   # 增大批处理（需要更多GPU内存）
--batch_size 128   # 减小批处理（节省GPU内存）
```

### 2. 模型选择优化

```python
# 只使用高性能细胞系模型
high_performance_cells = [
    'A549', 'MCF7', 'HT-29', 'HeLa', 'PC-3',
    'U-87-MG', 'SK-MEL-2', 'COLO-205'
]

ensemble.load_cell_line_models(high_performance_cells)
```

### 3. 数据采样优化

```bash
# 增加测试数据大小（更准确的评估）
--test_size 2000

# 减少测试数据大小（更快的评估）
--test_size 500
```

## 时间估算

### 训练时间（基于RTX 4090）

| 细胞系数量 | 训练时间 | 集成时间 | 总时间 |
|------------|----------|----------|--------|
| 5个 | 50-150分钟 | 5-10分钟 | 1-3小时 |
| 14个（免疫） | 140-420分钟 | 10-15分钟 | 3-7小时 |
| 50个 | 8-25小时 | 20-30分钟 | 9-26小时 |
| 323个（全部） | 54-162小时 | 1-2小时 | 55-164小时 |

### 内存需求

- **GPU内存**: 8GB+（推荐16GB+）
- **系统内存**: 32GB+（推荐64GB+）
- **存储空间**: 50GB+（所有模型和数据）

## 故障排除

### 1. 常见错误

#### CUDA内存不足
```bash
# 解决方案1: 减少批处理大小
--batch_size 128

# 解决方案2: 使用CPU
--device cpu

# 解决方案3: 分批加载模型
--cell_lines A549 MCF7 HT-29  # 只加载部分模型
```

#### MKL线程库冲突
```bash
# 设置环境变量
export MKL_SERVICE_FORCE_INTEL=1
```

#### 模型文件不存在
```bash
# 检查模型目录
ls result/model_train_results/PIC_*/PIC_*_model.pth

# 重新训练缺失的模型
python code/train_all_cell_lines.py --specific_cell_lines "A549" --device cuda:7 --overwrite
```

### 2. 性能调优

#### 集成性能不佳
1. **检查个体模型质量**: 查看各细胞系模型的训练结果
2. **调整投票策略**: 尝试加权投票或硬投票
3. **筛选高质量模型**: 只使用性能较好的细胞系模型

#### 预测速度慢
1. **减少模型数量**: 使用代表性细胞系子集
2. **优化硬件**: 使用更快的GPU或增加GPU内存
3. **批量预测**: 一次预测多个蛋白质

## 生物学应用

### 1. 泛癌症蛋白质重要性预测

```python
# 使用所有癌症细胞系模型
cancer_cells = [cl for cl in all_cell_lines if 'carcinoma' in meta_info[cl]['cancer_type'].lower()]
ensemble.load_cell_line_models(cancer_cells)
```

### 2. 组织特异性分析

```python
# 按组织类型分组
tissue_groups = {
    'lung': ['A549', 'NCI-H23', 'NCI-H1299', ...],
    'breast': ['MCF7', 'MDA-MB-231', 'T47D', ...],
    'colon': ['HT-29', 'HCT-116', 'SW620', ...]
}

for tissue, cells in tissue_groups.items():
    tissue_ensemble = PICCellEnsemble(...)
    tissue_ensemble.load_cell_line_models(cells)
    # 进行组织特异性预测
```

### 3. 药物靶点发现

```python
# 筛选高重要性蛋白质作为潜在药物靶点
high_importance_proteins = results_df[
    (results_df['y_prob'] >= 0.9) & 
    (results_df['y_pred'] == 1)
]

print(f"发现 {len(high_importance_proteins)} 个潜在药物靶点")
```

## 引用和参考

如果使用PIC-cell集成框架，请引用：

```bibtex
@article{pic_cell_ensemble,
  title={PIC-cell: A Multi-Cell-Line Ensemble Learning Framework for Protein Essentiality Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 联系和支持

- **项目主页**: [GitHub Repository]
- **文档**: 本指南和代码注释
- **问题反馈**: GitHub Issues
- **技术支持**: 查看日志文件和错误信息

---

**注意**: PIC-cell集成学习是一个计算密集型任务，建议在配置较高的服务器上运行完整流程。对于初次使用，推荐从测试流程开始。 