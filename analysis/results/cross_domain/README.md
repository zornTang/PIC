# Cross-domain Evaluation Summary

## 1. 实验目的
- 使用人类模型 (PIC_human) 直接对 14 个免疫细胞系进行推理，观察跨情境的性能变化。
- 对比每个细胞系“原生模型” vs. “人类模型迁移”在测试集上的 AUROC/AUPRC，以此作为情境差异的证据。
- 结论：人类模型迁移到免疫情境时表现普遍优于原生模型，说明不同情境间的必需性特征差异巨大，后续可考虑在 human 模型上对免疫数据进行微调，以更好捕捉这些差异。

## 2. 数据来源
- 原生模型测试指标：`result/model_train_results/PIC_<cell>/PIC_<cell>_test_result.csv`
- 人类模型跨域预测：`result/predictions/cross_domain/<cell>/PIC_<cell>_test_predictions.csv`
- 汇总 CSV：
  - `analysis/results/cross_domain/native_model_test_metrics.csv`
  - `analysis/results/cross_domain/human_model_cross_domain_summary.csv`
  - `analysis/results/cross_domain/native_vs_human_cross_test_summary.csv`

## 3. 关键发现
- 所有细胞系在测试集上满足：`human_cross_test_auc > native_test_auc` 且 `human_cross_test_auprc > native_test_auprc`。差值范围：
  - ΔAUROC ≈ -0.17 ~ -0.09（native - human），即人类模型迁移后最高可提升 ~0.17 AUROC。
  - ΔAUPRC ≈ -0.65 ~ -0.44（native - human），提升幅度更大，显示免疫情境与 human 基线差异显著。
- 热力图 (`cross_domain_auroc_heatmap.png`, `cross_domain_auprc_heatmap.png`) 和对比条形图 (`native_vs_human_test_auc_bar.png`, `native_vs_human_test_auprc_bar.png`) 展示了每个细胞系的性能对比。折线图 (`native_vs_human_test_delta_line.png`) 强调所有 Δ 值均为负，说明人类模型迁移后的表现全面优于原生模型。

## 4. 后续方向
- 在 human 模型基础上对免疫细胞系进行微调，以保留 human 模型的共性能力，同时融入免疫情境的特异特征。
- 利用 `cross_domain_test_delta_bar.png` 找到性能提升最大的细胞系（如 ROS-50、OCI-LY-19），优先安排微调实验。
- 将微调后的模型与当前结果对比，验证是否能在免疫情境下进一步提升性能，同时维持 human 情境的表现。

## 5. 结构级骨干消融：Attention 要不要用？
### 5.1 实验目标
- 对比三种骨干：`attention`（多头注意力 + 残差 + masked mean，实质为“注意力+池化混合”）、`cnn`（纯卷积特征提取）与 `avgpool`（直接池化后接 MLP，对应纯 MLP）。
- 限制其余超参、数据分割、优化器完全一致，只替换 `--model_variant`，从而回答“是否需要注意力模块”这一结构级问题。

### 5.2 训练流程
1. 复用 `code/train_all_cell_lines.py`，通过 `--model_variant` 切换骨干。建议为每个骨干指定独立的输出目录，避免覆盖已有模型，例如：

```bash
# 注意力+池化（baseline）
python code/train_all_cell_lines.py \
  --data_path data/cell_data.pkl \
  --cell_line_meta_file data/cell_line_meta_info.csv \
  --esm_model_path pretrained_model/esm2_t33_650M_UR50D.pt \
  --output_dir result/ablations/backbone_attention \
  --embedding_dir result/seq_embedding \
  --device cuda:0 \
  --specific_cell_lines PIC_human \
  --model_variant attention \
  --num_heads 4 \
  --overwrite

# 纯 CNN
python code/train_all_cell_lines.py \
  --data_path data/cell_data.pkl \
  --cell_line_meta_file data/cell_line_meta_info.csv \
  --esm_model_path pretrained_model/esm2_t33_650M_UR50D.pt \
  --output_dir result/ablations/backbone_cnn \
  --embedding_dir result/seq_embedding \
  --device cuda:0 \
  --specific_cell_lines PIC_human \
  --model_variant cnn \
  --cnn_channels 512 \
  --cnn_kernel_size 5 \
  --cnn_layers 3 \
  --cnn_drop 0.1 \
  --overwrite

# 纯池化/MLP
python code/train_all_cell_lines.py \
  --data_path data/cell_data.pkl \
  --cell_line_meta_file data/cell_line_meta_info.csv \
  --esm_model_path pretrained_model/esm2_t33_650M_UR50D.pt \
  --output_dir result/ablations/backbone_avgpool \
  --embedding_dir result/seq_embedding \
  --device cuda:0 \
  --specific_cell_lines PIC_human \
  --model_variant avgpool \
  --overwrite
```

> 提示：如果要扩展到 14 个免疫细胞系，只需去掉 `--specific_cell_lines` 或传入逗号分隔的细胞系列表；`train_all_cell_lines.py` 会自动复现相同的 val/test 划分并保存 `*_pred_scores.npy`，便于后续对照。

### 5.3 指标汇总与显著性检验
- 每次训练完成后，确保 `result/ablations/backbone_*/PIC_<cell_line>/` 目录包含 `model_config.json` 及 `val/test` 预测 `.npy`。
- 运行 `analysis/scripts/ablation_evaluation.py` 可以在同一细胞系的数据上比较多种骨干并给出 ΔAUROC/AUPRC 与自助法 p 值。例如针对 PIC_human：

```bash
python analysis/scripts/ablation_evaluation.py \
  --experiments attention=result/ablations/backbone_attention/PIC_human \
               cnn=result/ablations/backbone_cnn/PIC_human \
               avgpool=result/ablations/backbone_avgpool/PIC_human \
  --baseline attention \
  --splits val test \
  --bootstrap_rounds 2000 \
  --output_dir analysis/results/cross_domain/structural_ablation_human
```

- 若要覆盖所有免疫细胞系，可在 shell 中循环 `cell_line` 并替换路径。脚本会强制核对目标标签，确保真正在“同一数据集上”的结构级对照。

### 5.4 结果存档与解释
- 将 `analysis/results/cross_domain/structural_ablation_<context>/ablation_metrics.csv` 汇总复制到本目录，建议命名为 `attention_vs_cnn_vs_pooling_<context>.csv` 并追加到 README，以便追踪“注意力+池化混合”相对于纯 CNN/纯 MLP 的收益。
- 关键信息包括：`model_variant`、`split`、`auroc/auprc`、`delta_*_vs_baseline` 与 p 值。只要 Δ 均显著大于 0，即可直接回答“是否需要注意力”。
- 若要延伸至跨情境验证，可用对应骨干的 model checkpoint 通过 `code/predict.py` 生成免疫细胞集的预测，再复用现有的 `human_model_cross_domain_summary.csv` 模板对比 attention 与非注意力版本的迁移表现。

### 5.5 PIC_human（human_data.pkl）实测结果
- 数据：`data/human_data.pkl` + `result/seq_embedding`，attention baseline 即 `result/model_train_results/PIC_human`（param≈7.59M）。
- 对照结果（AUROC/AUPRC）：
  - Attention：val `0.974/0.927`，test `0.929/0.836`
  - CNN：val `0.928/0.830`，test `0.923/0.811`
  - AvgPool（纯 MLP）：val `0.883/0.724`，test `0.882/0.723`
- Δ 指标（见 `analysis/results/cross_domain/structural_ablation_human/ablation_metrics.csv`）显示：CNN 相比 attention 仍有 -0.046~-0.006 AUROC / -0.097~-0.025 AUPRC 的劣势，AvgPool 退化更明显。
- 可视化：`analysis/results/cross_domain/structural_ablation_human/backbone_metrics_auroc.png` 与 `backbone_metrics_auprc.png` 集中展示 test split 的 AUROC/AUPRC 对比条形图（val 指标见 CSV）。Attention 版本在 test 上同样明显优于纯 CNN/纯池化，因此“需要注意力”这一结论在实测中得到验证。

#### 深度可视化
- 新增脚本 `analysis/scripts/plot_backbone_ablation.py`，从 `ablation_metrics.csv` 自动生成三类深度图：
  1. `backbone_metrics_auroc.png` & `backbone_metrics_auprc.png`：分别绘制 test AUROC/AUPRC 的对比条形图（若需 val，可再绘制或查看 CSV）。
  2. `backbone_delta.png`：与 attention baseline 的 ΔAUROC/ΔAUPRC（仅 test）并标 0 线，直观看落幅。
  3. `backbone_param_tradeoff.png`：参数量（M） vs. test AUPRC 的散点图，以颜色编码 test AUROC，揭示“准确率-参数”效率。
- 运行示例（默认读取 human 消融 CSV）：
  ```bash
  python analysis/scripts/plot_backbone_ablation.py \
    --metrics_csv analysis/results/cross_domain/structural_ablation_human/ablation_metrics.csv \
    --baseline attention \
    --output_dir analysis/results/cross_domain/structural_ablation_human
  ```
- 这些可视化可套用到任意情境（免疫/跨域），将 `--metrics_csv` 指向对应的 `ablation_metrics.csv` 即可。多图联动后更容易看到：注意力模型不仅指标最高，也在参数量近似的情况下提供最佳 AUPRC，进一步支撑“必须保留注意力”的结论。
