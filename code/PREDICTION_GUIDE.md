# PIC蛋白质重要性预测使用指南

## 概述

本指南介绍如何使用训练好的PIC模型对新的蛋白质序列进行重要性预测，计算PES分数，并用于生物医学研究。

## 文件说明

- `predict.py`: 完整的预测脚本，支持单个和批量预测
- `predict_example.py`: 使用示例和教程
- `PREDICTION_GUIDE.md`: 本使用指南

## 快速开始

### 1. 环境准备

确保已安装所需依赖：
```bash
# 激活PIC环境
conda activate PIC

# 检查必要的包
python -c "import torch, pandas, numpy; from esm import pretrained; print('All dependencies available')"
```

### 2. 模型文件

确保有以下文件：
- 训练好的PIC模型：`./result/model_train_results/PIC_human/PIC_human_model.pth`
- ESM2预训练模型：`./pretrained_model/esm2_t33_650M_UR50D.pt`

### 3. 基本使用

#### 单个蛋白质预测
```python
from predict import ProteinPredictor

# 初始化预测器
predictor = ProteinPredictor(
    model_path="./result/model_train_results/PIC_human/PIC_human_model.pth",
    esm_model_path="./pretrained_model/esm2_t33_650M_UR50D.pt",
    device='cuda:7'
)

# 预测单个蛋白质
sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQ..."
probability = predictor.predict_single_protein(sequence)
pes_score = predictor.calculate_pes(probability)

print(f"重要性概率: {probability:.4f}")
print(f"PES分数: {pes_score:.4f}")
```

#### 批量预测（从FASTA文件）
```bash
python predict.py \
    --model_path ./result/model_train_results/PIC_human/PIC_human_model.pth \
    --input_fasta your_proteins.fasta \
    --output_file results.csv \
    --biomarker_analysis \
    --pes_threshold 2.0
```

## 详细功能

### 1. PES分数计算

PES (Protein Essentiality Score) = -log10(1 - probability)

**分数解释：**
- PES < 1.0: 低重要性 (概率 < 0.9)
- 1.0 ≤ PES < 2.0: 中等重要性 (0.9 ≤ 概率 < 0.99)
- 2.0 ≤ PES < 3.0: 高重要性 (0.99 ≤ 概率 < 0.999)
- PES ≥ 3.0: 极高重要性 (概率 ≥ 0.999)

### 2. 生物标志物分析

```python
# 分析潜在生物标志物
biomarkers_df = predictor.analyze_biomarkers(results_df, pes_threshold=2.0)

# 生物标志物等级
# - Low: PES 0-1.0
# - Medium: PES 1.0-2.0  
# - High: PES 2.0-3.0
# - Very High: PES > 3.0
```

### 3. 不同层次的模型

```bash
# 人类水平预测
python predict.py --model_path ./result/model_train_results/PIC_human/PIC_human_model.pth ...

# 小鼠水平预测  
python predict.py --model_path ./result/model_train_results/PIC_mouse/PIC_mouse_model.pth ...

# 细胞系水平预测 (A549)
python predict.py --model_path ./result/model_train_results/PIC_A549/PIC_A549_model.pth ...
```

## 输出结果

### 预测结果CSV文件包含：

| 列名 | 说明 |
|------|------|
| protein_id | 蛋白质ID |
| sequence | 蛋白质序列 |
| sequence_length | 序列长度 |
| essentiality_probability | 重要性概率 (0-1) |
| PES_score | 蛋白质重要性分数 |
| prediction | 预测结果 (Essential/Non-essential) |
| confidence | 置信度 (High/Medium/Low) |

### 生物标志物分析结果：

| 列名 | 说明 |
|------|------|
| biomarker_grade | 生物标志物等级 |
| 其他列 | 与预测结果相同 |

## 生物医学应用

### 1. 癌症研究

```python
# 识别癌症相关的重要蛋白质
cancer_proteins = results_df[
    (results_df['PES_score'] >= 2.0) & 
    (results_df['confidence'] == 'High')
]

# 按重要性排序
cancer_proteins = cancer_proteins.sort_values('PES_score', ascending=False)
```

### 2. 药物靶点发现

```python
# 筛选潜在药物靶点
drug_targets = results_df[
    (results_df['PES_score'] >= 2.5) & 
    (results_df['prediction'] == 'Essential')
]
```

### 3. 预后标志物

```python
# 识别预后相关的生物标志物
prognostic_markers = biomarkers_df[
    biomarkers_df['biomarker_grade'].isin(['High', 'Very High'])
]
```

## 示例用法

### 运行示例脚本
```bash
cd code
python predict_example.py
```

### 命令行预测示例
```bash
# 创建示例FASTA文件
cat > example.fasta << EOF
>P53_HUMAN
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
>BRCA1_HUMAN  
MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQ
EOF

# 运行预测
python predict.py \
    --model_path ./result/model_train_results/PIC_human/PIC_human_model.pth \
    --input_fasta example.fasta \
    --output_file example_results.csv \
    --biomarker_analysis \
    --pes_threshold 2.0
```

## 性能优化

### 1. GPU内存优化
```bash
# 如果遇到GPU内存不足，可以：
# 1. 减少批处理大小
# 2. 使用CPU预测
python predict.py --device cpu ...

# 3. 分批处理大文件
split -l 100 large_proteins.fasta batch_
```

### 2. 并行处理
```python
# 对于大量蛋白质，可以考虑并行处理
from multiprocessing import Pool

def predict_batch(fasta_file):
    # 预测逻辑
    pass

# 并行处理多个文件
with Pool(4) as p:
    results = p.map(predict_batch, fasta_files)
```

## 常见问题

### Q1: 模型文件找不到
**A:** 确保已完成模型训练，或下载预训练模型

### Q2: CUDA内存不足
**A:** 使用`--device cpu`或减少序列长度

### Q3: 预测结果置信度低
**A:** 检查序列质量，过短或过长的序列可能影响预测准确性

### Q4: 如何解释PES分数？
**A:** PES分数越高表示蛋白质越重要，≥2.0通常认为是高重要性

## 高级功能

### 1. 注意力权重可视化
```python
probability, attention_weights = predictor.predict_single_protein(
    sequence, return_attention=True
)

# 可视化注意力权重
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0], cmap='hot')
plt.title('Attention Weights')
plt.show()
```

### 2. 自定义阈值分析
```python
# 不同阈值下的生物标志物数量
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
for threshold in thresholds:
    biomarkers = predictor.analyze_biomarkers(results_df, threshold)
    print(f"PES >= {threshold}: {len(biomarkers)} biomarkers")
```

### 3. 结果统计分析
```python
import matplotlib.pyplot as plt
import seaborn as sns

# PES分数分布
plt.figure(figsize=(10, 6))
sns.histplot(results_df['PES_score'], bins=50)
plt.xlabel('PES Score')
plt.ylabel('Count')
plt.title('Distribution of PES Scores')
plt.show()

# 重要性概率vs序列长度
plt.figure(figsize=(10, 6))
plt.scatter(results_df['sequence_length'], results_df['essentiality_probability'])
plt.xlabel('Sequence Length')
plt.ylabel('Essentiality Probability')
plt.title('Protein Length vs Essentiality')
plt.show()
```

## 联系方式

如有问题或建议，请联系：
- 项目主页：https://github.com/KangBoming/PIC
- Web服务器：http://www.cuilab.cn/pic 