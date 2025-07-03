#!/usr/bin/env python3
"""
PIC-cell 集成学习框架
使用软投票策略聚合所有细胞系特异性PIC模型的预测结果
"""

import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加模块路径
sys.path.append('code')
from module.PIC import PIC
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


class PICCellEnsemble:
    """PIC-cell 集成模型类"""
    
    def __init__(self, model_dir: str, esm_model_path: str, device: str = 'cuda:7'):
        """
        初始化集成模型
        
        Args:
            model_dir: 细胞系模型目录
            esm_model_path: ESM2预训练模型路径
            device: 计算设备
        """
        self.model_dir = model_dir
        self.esm_model_path = esm_model_path
        self.device = device
        self.models = {}
        self.cell_lines = []
        self.weights = {}
        
        # 加载ESM2模型
        print("Loading ESM2 model...")
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_path)
        self.esm_model.eval()
        if torch.cuda.is_available():
            self.esm_model = self.esm_model.to(device)
        
        print("ESM2 model loaded successfully!")
    
    def load_cell_line_models(self, cell_lines: Optional[List[str]] = None):
        """
        加载所有可用的细胞系模型
        
        Args:
            cell_lines: 指定要加载的细胞系列表，None表示加载所有可用模型
        """
        print("Loading cell line specific models...")
        
        # 获取所有可用的模型
        available_models = []
        for model_folder in os.listdir(self.model_dir):
            if model_folder.startswith('PIC_'):
                cell_line = model_folder.replace('PIC_', '')
                model_path = os.path.join(self.model_dir, model_folder, f'PIC_{cell_line}_model.pth')
                if os.path.exists(model_path):
                    available_models.append((cell_line, model_path))
        
        # 筛选要加载的模型
        if cell_lines is not None:
            available_models = [(cl, path) for cl, path in available_models if cl in cell_lines]
        
        print(f"Found {len(available_models)} available models")
        
        # 加载模型
        loaded_count = 0
        for cell_line, model_path in available_models:
            try:
                model = PIC(
                    input_shape=1280,
                    hidden_units=320,
                    device=self.device,
                    linear_drop=0.1,
                    attn_drop=0.3,
                    output_shape=1
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                self.models[cell_line] = model
                self.cell_lines.append(cell_line)
                loaded_count += 1
                
                if loaded_count % 50 == 0:
                    print(f"Loaded {loaded_count} models...")
                    
            except Exception as e:
                print(f"Failed to load model for {cell_line}: {e}")
        
        print(f"Successfully loaded {len(self.models)} cell line models")
        print(f"Cell lines: {self.cell_lines[:10]}{'...' if len(self.cell_lines) > 10 else ''}")
    
    def extract_sequence_embedding(self, sequence: str, max_length: int = 1000):
        """
        提取蛋白质序列嵌入
        
        Args:
            sequence: 蛋白质序列
            max_length: 最大序列长度
            
        Returns:
            feature: 序列嵌入特征
            start_padding_idx: 填充开始位置
        """
        # 清理序列
        sequence = sequence.replace('*', '').upper()
        
        # 准备数据
        data = [("protein", sequence)]
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        batch_tokens = batch_tokens.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            representations = results["representations"][33]
            
            # 获取序列特征（去除BOS和EOS token）
            sequence_length = len(sequence)
            truncate_len = min(max_length, sequence_length)
            feature = representations[0, 1:truncate_len + 1]  # 去除BOS token
            
            # 处理填充
            if feature.shape[0] < max_length:
                pad_length = max_length - feature.shape[0]
                zero_features = torch.zeros((pad_length, 1280)).to(self.device)
                feature = torch.cat([feature, zero_features])
                start_padding_idx = torch.tensor(feature.shape[0] - pad_length).to(self.device)
            else:
                feature = feature[:max_length]
                start_padding_idx = torch.tensor(-1).to(self.device)
        
        return feature, start_padding_idx
    
    def predict_single_protein(self, sequence: str, voting_strategy: str = 'soft', 
                             return_individual: bool = False):
        """
        使用集成模型预测单个蛋白质
        
        Args:
            sequence: 蛋白质序列
            voting_strategy: 投票策略 ('soft', 'hard', 'weighted')
            return_individual: 是否返回每个模型的个体预测
            
        Returns:
            ensemble_prediction: 集成预测结果
            individual_predictions: 个体预测结果 (可选)
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded. Please call load_cell_line_models() first.")
        
        # 提取序列特征
        feature, start_padding_idx = self.extract_sequence_embedding(sequence)
        feature = feature.unsqueeze(0)
        start_padding_idx = start_padding_idx.unsqueeze(0)
        
        # 收集所有模型的预测
        predictions = []
        probabilities = []
        individual_results = {}
        
        with torch.no_grad():
            for cell_line, model in self.models.items():
                try:
                    logits = model(feature, start_padding_idx)
                    prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    pred = 1 if prob > 0.5 else 0
                    
                    predictions.append(pred)
                    probabilities.append(prob)
                    
                    if return_individual:
                        individual_results[cell_line] = {
                            'probability': prob,
                            'prediction': pred
                        }
                        
                except Exception as e:
                    print(f"Error in prediction for {cell_line}: {e}")
                    continue
        
        if len(probabilities) == 0:
            raise ValueError("No successful predictions from any model")
        
        # 集成预测
        if voting_strategy == 'soft':
            # 软投票：平均概率
            ensemble_prob = np.mean(probabilities)
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
        elif voting_strategy == 'hard':
            # 硬投票：多数决定
            ensemble_pred = 1 if sum(predictions) > len(predictions) / 2 else 0
            ensemble_prob = sum(predictions) / len(predictions)
            
        elif voting_strategy == 'weighted':
            # 加权投票：使用预设权重
            if not self.weights:
                # 如果没有权重，回退到软投票
                ensemble_prob = np.mean(probabilities)
            else:
                weighted_probs = []
                total_weight = 0
                for i, cell_line in enumerate(self.cell_lines):
                    if cell_line in self.weights and i < len(probabilities):
                        weighted_probs.append(probabilities[i] * self.weights[cell_line])
                        total_weight += self.weights[cell_line]
                
                ensemble_prob = sum(weighted_probs) / total_weight if total_weight > 0 else np.mean(probabilities)
            
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
        
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
        
        result = {
            'ensemble_probability': ensemble_prob,
            'ensemble_prediction': ensemble_pred,
            'num_models': len(probabilities),
            'voting_strategy': voting_strategy
        }
        
        if return_individual:
            result['individual_predictions'] = individual_results
            result['individual_probabilities'] = probabilities
        
        return result
    
    def evaluate_ensemble(self, test_data: pd.DataFrame, voting_strategy: str = 'soft',
                         ground_truth_column: str = 'label'):
        """
        评估集成模型性能
        
        Args:
            test_data: 测试数据，包含序列和真实标签
            voting_strategy: 投票策略
            ground_truth_column: 真实标签列名
            
        Returns:
            evaluation_results: 评估结果
        """
        print(f"Evaluating ensemble model with {voting_strategy} voting...")
        
        y_true = []
        y_pred = []
        y_prob = []
        
        for idx, row in test_data.iterrows():
            sequence = row['sequence']
            true_label = row[ground_truth_column]
            
            try:
                result = self.predict_single_protein(sequence, voting_strategy)
                
                y_true.append(true_label)
                y_pred.append(result['ensemble_prediction'])
                y_prob.append(result['ensemble_probability'])
                
                if (idx + 1) % 100 == 0:
                    print(f"Evaluated {idx + 1}/{len(test_data)} proteins...")
                    
            except Exception as e:
                print(f"Error evaluating protein {idx}: {e}")
                continue
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob),
            'num_samples': len(y_true),
            'voting_strategy': voting_strategy
        }
        
        print(f"\nEnsemble Model Evaluation Results ({voting_strategy} voting):")
        print("-" * 60)
        for metric, value in metrics.items():
            if metric not in ['num_samples', 'voting_strategy']:
                print(f"{metric.upper()}: {value:.4f}")
        print(f"Number of samples: {metrics['num_samples']}")
        
        return metrics, y_true, y_pred, y_prob
    
    def compare_voting_strategies(self, test_data: pd.DataFrame, 
                                ground_truth_column: str = 'label'):
        """
        比较不同投票策略的性能
        
        Args:
            test_data: 测试数据
            ground_truth_column: 真实标签列名
            
        Returns:
            comparison_results: 比较结果
        """
        strategies = ['soft', 'hard']
        if self.weights:
            strategies.append('weighted')
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Evaluating {strategy.upper()} voting strategy")
            print(f"{'='*60}")
            
            metrics, y_true, y_pred, y_prob = self.evaluate_ensemble(
                test_data, strategy, ground_truth_column
            )
            
            results[strategy] = {
                'metrics': metrics,
                'predictions': {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }
        
        # 创建比较表格
        comparison_df = pd.DataFrame({
            strategy: result['metrics'] 
            for strategy, result in results.items()
        }).T
        
        print(f"\n{'='*60}")
        print("VOTING STRATEGIES COMPARISON")
        print(f"{'='*60}")
        print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']])
        
        return results, comparison_df
    
    def set_model_weights(self, weights: Dict[str, float]):
        """
        设置模型权重（用于加权投票）
        
        Args:
            weights: 细胞系名称到权重的映射
        """
        self.weights = weights
        print(f"Set weights for {len(weights)} models")
    
    def calculate_adaptive_weights(self, validation_data: pd.DataFrame,
                                 ground_truth_column: str = 'label'):
        """
        基于验证数据计算自适应权重
        
        Args:
            validation_data: 验证数据
            ground_truth_column: 真实标签列名
            
        Returns:
            weights: 计算得到的权重
        """
        print("Calculating adaptive weights based on individual model performance...")
        
        individual_performances = {}
        
        # 评估每个模型的个体性能
        for cell_line, model in self.models.items():
            y_true = []
            y_pred = []
            y_prob = []
            
            print(f"Evaluating {cell_line}...")
            
            for idx, row in validation_data.iterrows():
                sequence = row['sequence']
                true_label = row[ground_truth_column]
                
                try:
                    feature, start_padding_idx = self.extract_sequence_embedding(sequence)
                    feature = feature.unsqueeze(0)
                    start_padding_idx = start_padding_idx.unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(feature, start_padding_idx)
                        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                        pred = 1 if prob > 0.5 else 0
                    
                    y_true.append(true_label)
                    y_pred.append(pred)
                    y_prob.append(prob)
                    
                except Exception as e:
                    continue
            
            if len(y_true) > 0:
                # 使用F1分数作为权重计算基准
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                # 组合F1和AUC作为权重
                weight = (f1 + auc) / 2
                individual_performances[cell_line] = weight
            else:
                individual_performances[cell_line] = 0.0
        
        # 归一化权重
        total_weight = sum(individual_performances.values())
        if total_weight > 0:
            weights = {cl: w/total_weight for cl, w in individual_performances.items()}
        else:
            weights = {cl: 1.0/len(individual_performances) for cl in individual_performances}
        
        self.set_model_weights(weights)
        
        # 显示权重信息
        print("\nCalculated adaptive weights:")
        print("-" * 40)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for cell_line, weight in sorted_weights[:10]:
            print(f"{cell_line}: {weight:.4f}")
        if len(sorted_weights) > 10:
            print("...")
        
        return weights
    
    def save_ensemble_model(self, save_path: str):
        """
        保存集成模型
        
        Args:
            save_path: 保存路径
        """
        ensemble_info = {
            'cell_lines': self.cell_lines,
            'weights': self.weights,
            'model_dir': self.model_dir,
            'esm_model_path': self.esm_model_path,
            'device': self.device,
            'num_models': len(self.models)
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_info, f)
        
        print(f"Ensemble model info saved to: {save_path}")
    
    def load_ensemble_model(self, load_path: str):
        """
        加载集成模型
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            ensemble_info = pickle.load(f)
        
        self.cell_lines = ensemble_info['cell_lines']
        self.weights = ensemble_info['weights']
        
        # 重新加载模型
        self.load_cell_line_models(self.cell_lines)
        
        print(f"Ensemble model loaded from: {load_path}")
        print(f"Loaded {len(self.models)} models")


def create_test_dataset(data_path: str, sample_size: int = 1000, 
                       random_state: int = 42) -> pd.DataFrame:
    """
    创建测试数据集
    
    Args:
        data_path: 数据文件路径
        sample_size: 样本大小
        random_state: 随机种子
        
    Returns:
        test_data: 测试数据
    """
    print(f"Creating test dataset from {data_path}...")
    
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 选择一个参考细胞系作为标签（例如A549）
    reference_cell_line = 'A549'
    if reference_cell_line not in data.columns:
        # 如果A549不存在，选择第一个可用的细胞系
        cell_columns = [col for col in data.columns if col not in ['index', 'ID', 'sequence']]
        reference_cell_line = cell_columns[0] if cell_columns else None
    
    if reference_cell_line is None:
        raise ValueError("No cell line columns found in data")
    
    print(f"Using {reference_cell_line} as reference cell line for testing")
    
    # 筛选有标签的数据
    valid_data = data[data[reference_cell_line].notna()].copy()
    
    # 采样
    if len(valid_data) > sample_size:
        test_data = valid_data.sample(n=sample_size, random_state=random_state)
    else:
        test_data = valid_data
    
    # 准备测试数据格式
    test_df = pd.DataFrame({
        'protein_id': test_data['ID'],
        'sequence': test_data['sequence'],
        'label': test_data[reference_cell_line].astype(int)
    })
    
    print(f"Created test dataset with {len(test_df)} samples")
    print(f"Label distribution: {test_df['label'].value_counts().to_dict()}")
    
    return test_df


def main():
    parser = argparse.ArgumentParser(description='PIC-cell Ensemble Learning Framework')
    
    # 基本参数
    parser.add_argument('--model_dir', type=str, default='result/model_train_results',
                       help='Directory containing trained cell line models')
    parser.add_argument('--esm_model_path', type=str, 
                       default='pretrained_model/esm2_t33_650M_UR50D.pt',
                       help='Path to ESM2 pretrained model')
    parser.add_argument('--data_path', type=str, default='data/cell_data.pkl',
                       help='Path to cell line data')
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='Device to use for inference')
    
    # 评估参数
    parser.add_argument('--test_size', type=int, default=1000,
                       help='Size of test dataset')
    parser.add_argument('--validation_size', type=int, default=500,
                       help='Size of validation dataset for weight calculation')
    parser.add_argument('--voting_strategy', type=str, default='soft',
                       choices=['soft', 'hard', 'weighted', 'all'],
                       help='Voting strategy to use')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='result/ensemble_results',
                       help='Output directory for results')
    parser.add_argument('--save_ensemble', action='store_true',
                       help='Save ensemble model')
    
    # 其他参数
    parser.add_argument('--cell_lines', type=str, nargs='+',
                       help='Specific cell lines to include (default: all available)')
    parser.add_argument('--adaptive_weights', action='store_true',
                       help='Calculate adaptive weights based on validation performance')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, f'ensemble_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print("="*80)
    print("PIC-cell Ensemble Learning Framework")
    print("="*80)
    
    # 初始化集成模型
    ensemble = PICCellEnsemble(
        model_dir=args.model_dir,
        esm_model_path=args.esm_model_path,
        device=args.device
    )
    
    # 加载模型
    ensemble.load_cell_line_models(args.cell_lines)
    
    if len(ensemble.models) == 0:
        print("No models loaded. Please train cell line models first.")
        return
    
    # 创建测试数据
    test_data = create_test_dataset(args.data_path, args.test_size)
    
    # 计算自适应权重（如果需要）
    if args.adaptive_weights:
        validation_data = create_test_dataset(args.data_path, args.validation_size, random_state=123)
        ensemble.calculate_adaptive_weights(validation_data)
    
    # 评估集成模型
    if args.voting_strategy == 'all':
        # 比较所有投票策略
        results, comparison_df = ensemble.compare_voting_strategies(test_data)
        
        # 保存比较结果
        comparison_df.to_csv(os.path.join(args.output_dir, 'voting_strategies_comparison.csv'))
        
    else:
        # 评估指定策略
        metrics, y_true, y_pred, y_prob = ensemble.evaluate_ensemble(test_data, args.voting_strategy)
        
        # 保存结果
        results_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        results_df.to_csv(os.path.join(args.output_dir, f'predictions_{args.voting_strategy}.csv'), index=False)
        
        # 保存评估指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.output_dir, f'metrics_{args.voting_strategy}.csv'), index=False)
    
    # 保存集成模型
    if args.save_ensemble:
        ensemble_save_path = os.path.join(args.output_dir, 'pic_cell_ensemble.pkl')
        ensemble.save_ensemble_model(ensemble_save_path)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main() 