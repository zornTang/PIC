# PIC (Protein Importance Calculator)

中性粒细胞蛋白质必需性预测项目，基于ESM2蛋白质语言模型实现人类层面和免疫细胞层面的双层预测框架，通过四组分类分析发现细胞类型特异性的蛋白质必需性机制。

## 🚀 快速开始

### 完整分析流程
```bash
# 1. 数据处理 (如需重新处理原始数据)
cd data_processing/scripts
python extract_gencode_proteins.py
python get_uniprot_annotations.py
python integrate_gencode_uniprot.py

# 2. 必需性分析 (使用现有数据)
cd ../../analysis/scripts
python neutrophil_comprehensive_analysis.py
```

### 查看结果
```bash
# 分析结果
ls analysis/results/

# 四组分类数据
head analysis/data/neutrophil_four_group_classification.csv
```

## 📁 项目结构

```
PIC/
├── README.md                    # 项目总览 (本文档)
├── data_processing/            # 🔧 数据处理模块
│   ├── scripts/               # 数据处理脚本
│   ├── raw_data/              # 原始数据文件
│   ├── processed_data/        # 处理后的数据
│   └── outputs/               # 处理结果输出
├── analysis/                   # 📊 分析模块
│   ├── scripts/               # 分析脚本
│   ├── data/                  # 分析用数据
│   └── results/               # 分析结果
├── references/                 # 📚 参考文献
├── code/                      # 传统核心源代码
├── data/                      # 传统训练数据
└── result/                    # 传统结果文件
```

## 🔬 核心创新

### 四组分类分析框架
- **人类特异必需**: 人类模型预测必需 + 免疫模型预测非必需
- **免疫特异必需**: 免疫模型预测必需 + 人类模型预测非必需
- **共同必需**: 两个模型都预测必需
- **共同非必需**: 两个模型都预测非必需

## 📊 主要发现

### 四组分类结果 (5,152个蛋白质)
- **共同非必需**: 3,559个 (69.1%) - 辅助功能、冗余通路
- **人类特异必需**: 1,115个 (21.6%) - 基础细胞功能、转录调节
- **免疫特异必需**: 327个 (6.3%) - 免疫应答、细胞激活
- **共同必需**: 151个 (2.9%) - 核心生存机制

### 生物学意义
| 蛋白质组 | 主要功能 | 临床意义 |
|---------|---------|----------|
| 人类特异必需 | 转录调节、表观遗传、基础代谢 | 广谱治疗靶点 |
| 免疫特异必需 | 组蛋白调节、细胞周期、抗氧化 | 精准免疫调节 |
| 共同必需 | 核糖体、基础代谢、能量产生 | 避免作为药物靶点 |
| 共同非必需 | 辅助功能、冗余通路 | 安全的干预靶点 |

## 📚 模块说明

- **[数据处理模块](data_processing/README.md)**: GENCODE + UniProt数据整合流程
- **[分析模块](analysis/README.md)**: 四组分类分析与可视化
- **[分析框架](analysis/comprehensive_analysis_framework.md)**: 多维度分析路线图

## 🎓 学术价值

### 发表潜力
- **方法学论文**: 四组分类分析框架
- **应用研究**: 中性粒细胞功能机制
- **综述文章**: 细胞类型特异性必需性

### 研究影响
- **系统生物学**: 多层次生物学信息整合
- **药物发现**: 靶点优先级排序新方法
- **精准医学**: 个体化治疗策略指导

---

*更新时间: 2025年1月15日*
*PIC项目 - 中性粒细胞蛋白质必需性预测与分析系统*
