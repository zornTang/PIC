# PIC 项目简化结构说明

## 项目结构优化

根据"训练独立、预测集成"的原则，项目结构已经简化：

### 核心文件结构
```
PIC/
├── code/
│   ├── main.py                    # 单个细胞系模型训练
│   ├── train_all_cell_lines.py    # 批量训练多个细胞系
│   ├── predict.py                 # 预测脚本（支持单模型和集成模式）
│   ├── embedding.py               # 序列嵌入提取
│   ├── visualize_protein_predictions.py  # 结果可视化
│   ├── PREDICTION_GUIDE.md        # 预测使用指南
│   └── module/                    # 核心模块
│       ├── PIC.py                 # PIC模型定义
│       ├── load_dataset.py        # 数据加载
│       ├── loss_func.py           # 损失函数
│       └── earlystopping.py       # 早停机制
├── data/                          # 数据文件
├── pretrained_model/              # 预训练模型
├── result/                        # 训练结果和预测结果
├── README.md                      # 项目说明
└── environment.yml                # 环境配置
```

### 已删除的文件
- `ensemble_pic_cell.py` - 独立的集成学习评估框架（功能已集成到predict.py）
- `train_and_ensemble_all_cells.py` - 集成训练流程脚本
- `analyze_immune_cell_lines.py` - 免疫细胞系分析脚本
- `check_environment.py` - 环境检查脚本
- `cleanup_project.sh` - 项目清理脚本
- `monitor_pic_cell.sh` - 监控脚本

### 使用流程

#### 1. 训练阶段（独立训练）
```bash
# 训练单个细胞系
python code/main.py --label_name "A549" --data_path data/cell_data.pkl --feature_dir result/seq_embedding --save_path result/model_train_results

# 批量训练多个细胞系
python code/train_all_cell_lines.py --specific_cell_lines "A549,HeLa,MCF7" --device cuda:7
```

#### 2. 预测阶段（可选集成）
```bash
# 单模型预测
python code/predict.py --model_path result/model_train_results/PIC_A549/PIC_A549_model.pth --input_fasta proteins.fasta

# 集成模型预测
python code/predict.py --ensemble_mode --model_dir result/model_train_results --input_fasta proteins.fasta
```

### 优化效果

1. **简化训练流程**：保持各细胞系模型独立训练，避免复杂的集成训练逻辑
2. **灵活预测方式**：支持单模型和集成模式，用户可根据需求选择
3. **代码维护性**：减少冗余代码，提高可维护性
4. **清晰的职责分离**：训练和预测功能明确分离

### 核心功能保留

- ✅ 单个细胞系模型训练
- ✅ 批量训练多个细胞系  
- ✅ 单模型预测
- ✅ 集成模型预测（软投票/硬投票）
- ✅ PES分数计算
- ✅ 生物标志物分析
- ✅ 结果可视化

这种简化结构既保持了功能完整性，又提高了代码的可维护性和使用便利性。 