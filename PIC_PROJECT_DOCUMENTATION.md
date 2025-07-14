# PIC (Protein Importance Calculator) 项目完整文档

## 📋 项目概述

PIC是一个基于ESM2蛋白质语言模型的深度学习系统，用于预测蛋白质在不同生物学层面的必需性。本项目实现了人类层面和免疫细胞层面的双层预测框架，并完成了对中性粒细胞相关蛋白质的全面分析。

### 🎯 核心功能
- **双层面预测**: 人类层面 + 免疫细胞层面必需性预测
- **集成学习**: 14个免疫细胞系模型的加权集成
- **对比分析**: 两种预测结果的全面比较和可视化
- **功能注释**: 蛋白质功能分类和生物学意义解释

---

## 📁 项目结构

```
PIC/
├── 📂 code/                    # 核心源代码
│   ├── embedding.py            # 序列嵌入提取
│   ├── main.py                 # 模型训练
│   ├── predict.py              # 预测功能
│   ├── train_all_cell_lines.py # 批量训练
│   ├── visualize_protein_predictions.py # 可视化
│   └── module/                 # 模块代码
│
├── 📂 data/                    # 训练数据
│   ├── human_data.pkl          # 人类层面数据
│   ├── cell_data.pkl           # 细胞系数据
│   ├── cell_line_meta_info.csv # 细胞系元信息
│   └── mouse_data.pkl          # 小鼠数据
│
├── 📂 pretrained_model/        # 预训练模型
│   ├── esm2_t33_650M_UR50D.pt  # ESM2主模型
│   └── esm2_t33_650M_UR50D-contact-regression.pt
│
├── 📂 result/                  # 所有结果文件
│   ├── seq_embedding/          # 序列嵌入 (65,057个文件)
│   ├── model_train_results/    # 训练模型 (15个)
│   ├── predictions/            # 预测结果
│   └── neutrophil_analysis/    # 完整分析结果
│       ├── visualizations/     # 可视化图表
│       ├── reports/           # 分析报告
│       └── data/              # 分析数据
│
├── 📂 logs/                    # 日志目录
├── 🛠️ 核心工具
│   ├── compare_predictions.py     # 预测对比分析
│   ├── check_project_status.py    # 项目状态检查
│   ├── run_training_workflow.sh   # 训练阶段工作流程
│   └── run_complete_workflow.sh   # 完整流程执行
└── 📄 neutrophil_mane_proteins.fa # 中性粒细胞蛋白序列
```

---

## 🔄 完整工作流程

### 阶段1: 环境准备
```bash
# 激活虚拟环境
conda activate PIC

# 检查GPU可用性
nvidia-smi
```

### 阶段2: 序列嵌入提取
```bash
python code/embedding.py \
    --data_path data/human_data.pkl \
    --model pretrained_model/esm2_t33_650M_UR50D.pt \
    --output_dir result/seq_embedding \
    --device cuda:7
```

### 阶段3: 模型训练

#### 方法1: 一键训练（推荐）
```bash
# 使用默认参数执行完整训练流程
bash run_training_workflow.sh

# 自定义参数训练
bash run_training_workflow.sh cuda:7 64 15 1e-5 false
```

#### 方法2: 分步训练

##### 人类层面模型
```bash
python code/main.py \
    --data_path data/human_data.pkl \
    --feature_dir result/seq_embedding \
    --label_name human \
    --save_path result/model_train_results \
    --device cuda:7
```

##### 免疫细胞系模型
```bash
python code/train_all_cell_lines.py \
    --specific_cell_lines 'ARH-77,IM-9,KMS-11,L-363,LP-1,OCI-AML2,OCI-AML3,OCI-LY-19,OPM-2,ROS-50,RPMI-8226,SU-DHL-10,SU-DHL-5,SU-DHL-8' \
    --device cuda:7
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| GPU设备 | cuda:7 | 指定使用的GPU设备 |
| 批次大小 | 64 | 训练和推理的批次大小 |
| 训练轮数 | 15 | 模型训练的最大轮数 |
| 学习率 | 1e-5 | 优化器学习率 |
| 覆盖模式 | false | 是否覆盖已有的模型 |

### 阶段4: 预测执行

#### 人类层面预测
```bash
python code/predict.py \
    --model_path result/model_train_results/PIC_human/PIC_human_model.pth \
    --input_fasta neutrophil_mane_proteins.fa \
    --output_file result/predictions/neutrophil_human_predictions.csv \
    --device cuda:7
```

#### 免疫层面集成预测
```bash
python code/predict.py \
    --ensemble_mode \
    --model_dir result/model_train_results \
    --input_fasta neutrophil_mane_proteins.fa \
    --output_file result/predictions/neutrophil_immune_ensemble_predictions.csv \
    --device cuda:7
```

### 阶段5: 对比分析
```bash
python compare_predictions.py
```

---

## 🚀 快速开始

### 方法1: 完整训练+分析流程
```bash
# 1. 执行训练阶段（如果模型未训练）
bash run_training_workflow.sh

# 2. 执行预测和分析
bash run_complete_workflow.sh
```

### 方法2: 仅分析已有结果
```bash
# 1. 检查项目状态
python check_project_status.py

# 2. 运行预测分析
python compare_predictions.py

# 3. 查看结果
ls result/neutrophil_analysis/
```

### 方法3: 自定义训练参数
```bash
# 自定义GPU、批次大小等参数
bash run_training_workflow.sh cuda:6 128 20 2e-5 true
```

---

## 📊 主要发现

### 预测结果统计
| 指标 | 人类层面模型 | 免疫层面模型 | 差异 |
|------|-------------|-------------|------|
| 总蛋白质数 | 1,249 | 1,249 | - |
| 必需蛋白质 | 255 (20.4%) | 146 (11.7%) | +8.7% |
| 预测一致性 | - | - | 75.9% |
| 分数相关性 | - | - | r=0.116 |

### 功能模式差异
| 功能类别 | 人类特异性 | 免疫特异性 | 共识蛋白质 |
|----------|------------|------------|------------|
| 转录调节 | 3.9% | 1.0% | 26.0% |
| 组蛋白相关 | 2.0% | 10.4% | 4.0% |
| 代谢酶 | 6.3% | 8.3% | 42.0% |
| 抗氧化酶 | - | 8.3% | 4.0% |

### 关键蛋白质发现

#### 人类特异性必需蛋白质
- **CREB5**: 转录调节因子 - cAMP反应元件结合蛋白
- **SOX6**: 转录因子 - SRY相关HMG-box蛋白
- **SMCHD1**: 表观遗传调节 - 结构维持染色体蛋白

#### 免疫特异性必需蛋白质
- **H2BC4/H2BC11/H2BC18**: 组蛋白 - 核心组蛋白H2B
- **SOD2**: 抗氧化酶 - 超氧化物歧化酶2
- **HYCC2**: 细胞周期调节 - 细胞周期检查点蛋白

#### 共识必需蛋白质
- **RPS4Y2**: 核糖体蛋白 - 40S核糖体蛋白S4Y2
- **NAMPT**: 代谢酶 - 烟酰胺磷酸核糖转移酶
- **TLE3**: 转录共抑制因子 - Groucho家族

---

## 🔬 生物学意义

### 人类层面模型特点
- 反映复杂的基因调控网络需求
- 转录调节因子和表观遗传调节蛋白重要性突出
- 适合作为广谱治疗靶点

### 免疫层面模型特点
- 组蛋白相关蛋白质在免疫细胞激活中关键
- 抗氧化酶应对免疫应答中的氧化应激
- 适合作为精准免疫调节靶点

### 临床应用价值
- **药物靶点发现**: 不同层面的特异性靶点
- **疾病研究**: 为自身免疫疾病和癌症研究提供指导
- **个性化医疗**: 上下文特异性的精准治疗

---

## 📈 分析结果

### 可视化图表
1. **概览分析** (`01_overview_comparison.png`)
   - PES分数相关性分析
   - 预测一致性分析
   - 分数分布对比

2. **详细分析** (`02_detailed_analysis.png`)
   - 混淆矩阵热图
   - 预测组合分析
   - 功能富集分析

3. **生物标志物分析** (`03_biomarker_analysis.png`)
   - 高置信度蛋白质分布
   - Bland-Altman分析
   - 预测置信度分布

4. **功能分析** (`04_functional_analysis.png`)
   - 功能分类饼图
   - 顶级蛋白质排名
   - 生物学意义总结

### 分析报告
- **详细报告**: `result/neutrophil_analysis/reports/enhanced_comparison_report.md`
- **数据文件**: `result/neutrophil_analysis/data/prediction_comparison_results.csv`
- **可视化**: `result/neutrophil_analysis/visualizations/` (4个专业图表)

---

## 🛠️ 工具说明

### 核心工具
- **`compare_predictions.py`**: 主要分析功能，生成完整的对比分析
- **`check_project_status.py`**: 检查项目完整性和状态
- **`run_training_workflow.sh`**: 专门的训练阶段工作流程脚本
- **`run_complete_workflow.sh`**: 执行完整的工作流程

### 核心代码
- **`code/embedding.py`**: ESM2序列嵌入提取
- **`code/main.py`**: 单模型训练
- **`code/predict.py`**: 预测和集成预测
- **`code/train_all_cell_lines.py`**: 批量训练多个细胞系模型

---

## 📝 使用注意事项

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA支持的GPU
- 至少32GB内存

### 文件要求
- ESM2预训练模型文件
- 训练数据文件 (human_data.pkl, cell_data.pkl)
- 中性粒细胞蛋白序列文件

### 运行建议
- 使用GPU进行训练和预测
- 确保足够的磁盘空间存储结果
- 定期检查项目状态
- 长时间训练建议使用后台运行: `nohup bash run_training_workflow.sh > training.log 2>&1 &`

### 训练工作流程详细说明

#### 🔧 训练脚本使用

**基本语法:**
```bash
bash run_training_workflow.sh [GPU设备] [批次大小] [训练轮数] [学习率] [覆盖模式]
```

**使用示例:**
```bash
# 默认配置
bash run_training_workflow.sh

# 自定义GPU和批次大小
bash run_training_workflow.sh cuda:0 32

# 完全自定义配置
bash run_training_workflow.sh cuda:6 128 20 2e-5 true

# 后台运行
nohup bash run_training_workflow.sh > training.log 2>&1 &
```

#### ⏱️ 预计训练时间

| 阶段 | 预计耗时 | 说明 |
|------|----------|------|
| 序列嵌入提取 | 30-60分钟 | 约65,000个蛋白质序列 |
| 人类层面模型 | 15-30分钟 | 1个通用模型 |
| 免疫细胞系模型 | 3-6小时 | 14个特异性模型 |
| **总计** | **4-7小时** | 完整训练流程 |

#### 🎯 训练输出

**序列嵌入:**
- 位置: `result/seq_embedding/`
- 数量: ~65,000个 `.pt` 文件
- 大小: 每个文件约1-5MB

**训练模型:**
- 位置: `result/model_train_results/`
- 数量: 15个模型目录
- 内容: 模型文件(.pth) + 训练结果(.csv)

#### ⚠️ 故障排除

**常见问题及解决方案:**

1. **GPU内存不足**
   ```bash
   # 减小批次大小
   bash run_training_workflow.sh cuda:7 32
   ```

2. **磁盘空间不足**
   ```bash
   # 检查空间
   df -h
   # 清理临时文件
   find . -name "*.tmp" -delete
   ```

3. **训练中断恢复**
   ```bash
   # 检查已完成的模型
   python check_project_status.py
   # 继续训练（跳过已有模型）
   bash run_training_workflow.sh
   ```

---

## 🎯 项目成果

### 学术价值
- 建立了蛋白质必需性的双层面预测框架
- 发现了人类层面和免疫层面的功能模式差异
- 为蛋白质功能研究提供了新的分析视角

### 技术贡献
- 开发了完整的蛋白质必需性预测流程
- 建立了标准化的分析和可视化框架
- 创建了可重现的研究工作流程

### 实用价值
- 识别了潜在的治疗靶点
- 为疾病研究提供了指导
- 支持个性化医疗发展

---

## 📞 联系信息

如有问题或建议，请参考项目文档或使用提供的检查工具进行诊断。

---

*PIC项目 - 蛋白质必需性预测与分析系统*  
*版本: 1.0*  
*更新时间: 2024年7月*
