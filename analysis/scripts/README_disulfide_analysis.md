# 二硫键形成潜力分析模块

## 概述

`disulfide_bond_analysis.py` 是一个独立的Python模块，专门用于分析蛋白质序列中二硫键形成的潜力和数量。该模块从原始的 `compare_predictions.py` 文件中分离出来，提供了模块化的二硫键预测和分析功能。

## 主要功能

### 1. 二硫键形成概率预测
- **函数**: `predict_disulfide_bonds(sequence)`
- **功能**: 基于经验规则预测单个蛋白质序列的二硫键形成潜力
- **算法原理**:
  - 空间距离效应：相邻半胱氨酸更容易形成二硫键
  - 密度效应：半胱氨酸密度适中时形成概率更高
  - 长度效应：较短蛋白质更容易形成正确配对
  - 经验概率模型：基于统计规律

### 2. 批量数据分析
- **函数**: `add_disulfide_analysis(df)`
- **功能**: 为包含蛋白质序列的数据框添加二硫键分析列
- **输出列**:
  - `cys_count`: 半胱氨酸数量
  - `potential_disulfide_bonds`: 潜在二硫键数量
  - `disulfide_probability`: 二硫键形成概率
  - `cys_percentage`: 半胱氨酸百分比
  - `bond_distribution`: 分布模式 (none/clustered/mixed/distributed)

### 3. 统计分析
- **函数**: `analyze_disulfide_bonds(merged_df)`
- **功能**: 分析不同蛋白质组之间的二硫键特征差异
- **包括**: 描述性统计、显著性检验、分布模式分析

### 4. 可视化
- **函数**: `create_disulfide_bond_analysis_visualization(merged_df, human_specific, immune_specific)`
- **功能**: 创建包含6个子图的综合二硫键分析图表
- **图表类型**: 箱线图、饼图、统计显著性标注

### 5. 整体模式分析
- **函数**: `analyze_disulfide_patterns(df, sequence_col='sequence')`
- **功能**: 分析数据集的整体二硫键形成模式和统计特征

## 使用示例

### 基本用法

```python
from disulfide_bond_analysis import predict_disulfide_bonds, add_disulfide_analysis

# 1. 单个序列分析
sequence = "MKTIIALSYIFCLVFADYKDDDDKCRPVVKCCSCC"
result = predict_disulfide_bonds(sequence)
print(f"半胱氨酸数量: {result['cys_count']}")
print(f"潜在二硫键: {result['potential_disulfide_bonds']}")
print(f"形成概率: {result['disulfide_probability']:.3f}")

# 2. 数据框批量分析
import pandas as pd
df = pd.DataFrame({
    'protein_id': ['P1', 'P2', 'P3'],
    'sequence': [sequence1, sequence2, sequence3]
})
df_analyzed = add_disulfide_analysis(df)
```

### 高级分析

```python
from disulfide_bond_analysis import (
    analyze_disulfide_bonds,
    create_disulfide_bond_analysis_visualization,
    analyze_disulfide_patterns
)

# 比较分析
human_specific, immune_specific = analyze_disulfide_bonds(merged_df)

# 创建可视化
fig = create_disulfide_bond_analysis_visualization(
    merged_df, human_specific, immune_specific
)
fig.savefig('disulfide_analysis.png', dpi=300, bbox_inches='tight')

# 整体模式分析
stats, df_analyzed = analyze_disulfide_patterns(protein_df)
print(f"总蛋白质数: {stats['total_proteins']}")
print(f"含半胱氨酸蛋白质: {stats['proteins_with_cys']}")
print(f"分布模式: {stats['distribution_patterns']}")
```

## 算法详解

### 二硫键形成概率计算

算法考虑以下因素：

1. **基础概率**:
   - 偶数个半胱氨酸: 0.7
   - 奇数个半胱氨酸: 0.5

2. **距离因子**:
   - 理想距离: 3-50个氨基酸残基
   - 计算公式: `optimal_distances / total_distances`

3. **密度因子**:
   - 理想密度: 1-8% 半胱氨酸含量
   - 过高密度(>12%): 降低因子至0.8

4. **长度因子**:
   - 短蛋白质(<100aa): 因子1.1
   - 长蛋白质(>500aa): 因子0.9

5. **最终概率**:
   ```
   final_probability = min(
       base_probability × distance_factor × density_factor × length_factor,
       1.0
   )
   ```

### 分布模式分类

- **none**: 半胱氨酸数量 < 2
- **clustered**: 距离范围 < 20个氨基酸
- **mixed**: 包含距离 < 10个氨基酸的配对
- **distributed**: 其他情况

## 依赖库

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
```

## 输出说明

### 预测结果字典包含以下键值:

- `cys_count`: 半胱氨酸总数
- `cys_positions`: 半胱氨酸位置列表
- `potential_disulfide_bonds`: 潜在二硫键数量 (cys_count // 2)
- `disulfide_probability`: 形成概率 (0.0-1.0)
- `cys_percentage`: 半胱氨酸百分比
- `min_distance/max_distance/avg_distance`: 距离统计
- `bond_distribution`: 分布模式
- `distance_factor/density_factor`: 计算因子

## 注意事项

1. **序列格式**: 只处理标准20种氨基酸，自动过滤非标准字符
2. **概率模型**: 基于经验规则，实际结构可能有所不同
3. **计算复杂度**: O(n×m) 其中n为蛋白质数量，m为平均序列长度
4. **内存使用**: 大数据集时注意内存占用

## 集成使用

该模块可以轻松集成到现有的生物信息学分析流程中：

```python
# 在主分析脚本中导入
from disulfide_bond_analysis import add_disulfide_analysis, analyze_disulfide_bonds

# 添加到数据处理流程
df = add_disulfide_analysis(df)

# 进行统计分析
human_specific, immune_specific = analyze_disulfide_bonds(merged_df)
```

## 更新日志

- **v1.0**: 从 `compare_predictions.py` 分离出独立模块
- 包含完整的二硫键预测、分析和可视化功能
- 提供模块化接口和详细文档