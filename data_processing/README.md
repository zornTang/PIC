# 中性粒细胞蛋白质数据处理 (Neutrophil Protein Data Processing)

## 📁 目录结构

```
data_processing/
├── README.md                          # 本文档
├── scripts/                          # 数据处理脚本
│   ├── extract_gencode_proteins.py    # 从GENCODE提取蛋白质序列
│   ├── get_uniprot_annotations.py     # 获取UniProt功能注释
│   └── integrate_gencode_uniprot.py   # 整合GENCODE和UniProt数据
├── raw_data/                         # 原始数据文件
│   ├── gencode.v45.annotation.gtf.gz                    # GENCODE注释文件
│   ├── gencode.v45.pc_translations.fa.gz               # GENCODE蛋白质序列
│   ├── immune_cell_category_rna_neutrophil_Immune.tsv  # 免疫细胞RNA表达数据
│   └── neutrophil_genes_list.txt                       # 中性粒细胞基因列表
├── processed_data/                   # 处理后的数据
│   └── uniprot_annotations/          # UniProt注释数据
└── outputs/                          # 输出结果
    ├── neutrophil_proteins_output/   # 中性粒细胞蛋白质处理结果
    └── integrated_proteins_output/   # 整合后的蛋白质数据
```

## 🔄 数据处理流程

### 第一步：提取GENCODE蛋白质信息
```bash
cd scripts
python extract_gencode_proteins.py
```
**功能**: 从GENCODE GTF文件和FASTA文件中提取中性粒细胞相关的蛋白质信息

**输入文件**:
- `../raw_data/gencode.v45.annotation.gtf.gz`
- `../raw_data/gencode.v45.pc_translations.fa.gz`
- `../raw_data/neutrophil_genes_list.txt`

**输出**: `../outputs/neutrophil_proteins_output/`

### 第二步：获取UniProt功能注释
```bash
python get_uniprot_annotations.py
```
**功能**: 根据蛋白质ID获取UniProt数据库的功能注释信息

**输入**: 从第一步提取的蛋白质ID列表
**输出**: `../processed_data/uniprot_annotations/`

### 第三步：整合GENCODE和UniProt数据
```bash
python integrate_gencode_uniprot.py
```
**功能**: 将GENCODE结构信息与UniProt功能注释整合，生成完整的蛋白质数据集

**输入**:
- GENCODE提取结果
- UniProt注释数据

**输出**: `../outputs/integrated_proteins_output/`

## 📊 数据文件说明

### 原始数据 (raw_data/)
- **GENCODE GTF**: 基因注释文件，包含基因结构信息
- **GENCODE FASTA**: 蛋白质序列文件
- **免疫细胞RNA数据**: 中性粒细胞特异性表达数据
- **基因列表**: 中性粒细胞相关基因名单

### 处理后数据 (processed_data/)
- **UniProt注释**: 蛋白质功能、定位、结构域等注释信息

### 输出结果 (outputs/)
- **中性粒细胞蛋白质数据**: 提取的中性粒细胞特异性蛋白质
- **整合蛋白质数据**: 结合结构和功能信息的完整数据集

## 🔧 脚本功能详解

### extract_gencode_proteins.py
- 解析GENCODE GTF注释文件
- 提取蛋白质编码基因信息
- 匹配蛋白质序列
- 筛选中性粒细胞相关蛋白质

### get_uniprot_annotations.py
- 查询UniProt数据库API
- 获取蛋白质功能描述
- 收集亚细胞定位信息
- 提取蛋白质结构域数据

### integrate_gencode_uniprot.py
- 基于蛋白质ID匹配数据
- 整合序列和功能信息
- 生成统一的数据格式
- 输出分析就绪的数据集

## 📈 数据质量控制

### 匹配率统计
- GENCODE-UniProt ID匹配成功率
- 功能注释完整性检查
- 序列长度分布验证

### 数据验证
- 重复项检测和去除
- 数据格式一致性检查
- 缺失值处理策略

## 🚀 快速开始

```bash
# 完整的数据处理流程
cd data_processing/scripts

# 第1步：提取GENCODE数据
python extract_gencode_proteins.py

# 第2步：获取UniProt注释
python get_uniprot_annotations.py

# 第3步：整合数据
python integrate_gencode_uniprot.py

# 检查输出结果
ls ../outputs/integrated_proteins_output/
```

## 📋 依赖要求

```python
# 主要Python包
pandas >= 1.3.0
requests >= 2.25.0
gzip
json
pathlib
```

## ⚡ 性能优化

- **并行处理**: UniProt API查询使用多线程
- **缓存机制**: 避免重复下载相同注释
- **分块处理**: 大文件分批处理减少内存占用
- **错误重试**: 网络请求失败自动重试

## 📝 注意事项

1. **网络依赖**: UniProt注释需要网络连接
2. **处理时间**: 大规模数据处理可能需要数小时
3. **存储空间**: 确保有足够磁盘空间存储中间文件
4. **API限制**: 遵守UniProt API使用限制

## 🔗 相关链接

- [GENCODE数据库](https://www.gencodegenes.org/)
- [UniProt数据库](https://www.uniprot.org/)
- [分析脚本目录](../analysis/) - 后续分析流程

---

*更新时间: 2025年1月15日*