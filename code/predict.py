import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from module.PIC import PIC
from typing import List, Dict, Optional


class ProteinPredictor:
    """蛋白质重要性预测器 - 支持单模型和集成模型预测"""
    
    def __init__(self, model_path=None, esm_model_path='./pretrained_model/esm2_t33_650M_UR50D.pt', 
                 device='cuda:7', ensemble_mode=False, model_dir=None, cell_lines=None):
        """
        初始化预测器
        
        Args:
            model_path: 单个训练好的PIC模型路径（单模型模式）
            esm_model_path: ESM2预训练模型路径
            device: 计算设备
            ensemble_mode: 是否使用集成模式
            model_dir: 模型目录（集成模式）
            cell_lines: 细胞系列表（集成模式）
        """
        self.device = device
        self.esm_model_path = esm_model_path
        self.ensemble_mode = ensemble_mode
        
        # 加载ESM2模型
        print("Loading ESM2 model...")
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_path)
        self.esm_model.eval()
        if torch.cuda.is_available():
            self.esm_model = self.esm_model.to(device)
        
        if ensemble_mode:
            # 集成模式：加载多个模型
            print("Initializing ensemble mode...")
            self.models = {}
            self.cell_lines = cell_lines or [
                "ARH-77", "IM-9", "KMS-11", "L-363", "LP-1",
                "OCI-AML2", "OCI-AML3", "OCI-LY-19", "OPM-2",
                "ROS-50", "RPMI-8226", "SU-DHL-10", "SU-DHL-5", "SU-DHL-8"
            ]
            self.load_ensemble_models(model_dir)
        else:
            # 单模型模式
            print("Loading single PIC model...")
            self.pic_model = PIC(
                input_shape=1280,
                hidden_units=320,
                device=device,
                linear_drop=0.1,
                attn_drop=0.3,
                output_shape=1
            )
            self.pic_model.load_state_dict(torch.load(model_path, map_location=device))
            self.pic_model.to(device)
            self.pic_model.eval()
        
        print("Models loaded successfully!")
    
    def load_ensemble_models(self, model_dir):
        """加载集成模型"""
        if not model_dir:
            raise ValueError("model_dir is required for ensemble mode")
        
        print(f"Loading ensemble models from {model_dir}...")
        loaded_count = 0
        
        for cell_line in self.cell_lines:
            model_path = os.path.join(model_dir, f"PIC_{cell_line}", f"PIC_{cell_line}_model.pth")
            
            if os.path.exists(model_path):
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
                    loaded_count += 1
                    print(f"✓ Loaded {cell_line}")
                    
                except Exception as e:
                    print(f"✗ Failed to load {cell_line}: {e}")
            else:
                print(f"✗ Model not found: {model_path}")
        
        print(f"Successfully loaded {loaded_count}/{len(self.cell_lines)} models")
        
        if loaded_count == 0:
            raise ValueError("No models loaded successfully")
    
    def extract_sequence_embedding(self, sequence, max_length=1000):
        """
        提取单个蛋白质序列的嵌入特征
        
        Args:
            sequence: 蛋白质序列字符串
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
    
    def predict_single_protein(self, sequence, return_attention=False, voting_strategy='soft'):
        """
        预测单个蛋白质的重要性
        
        Args:
            sequence: 蛋白质序列
            return_attention: 是否返回注意力权重（仅单模型模式）
            voting_strategy: 投票策略（仅集成模式）：'soft', 'hard'
            
        Returns:
            prediction: 预测概率或集成结果
            attention_weights: 注意力权重（可选）
        """
        # 提取特征
        feature, start_padding_idx = self.extract_sequence_embedding(sequence)
        
        # 添加batch维度
        feature = feature.unsqueeze(0)
        start_padding_idx = start_padding_idx.unsqueeze(0)
        
        if self.ensemble_mode:
            # 集成预测
            return self._ensemble_predict(feature, start_padding_idx, voting_strategy)
        else:
            # 单模型预测
            with torch.no_grad():
                if return_attention:
                    logits, attention_weights = self.pic_model(feature, start_padding_idx, get_attention=True)
                    probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    return probability, attention_weights.cpu().numpy()
                else:
                    logits = self.pic_model(feature, start_padding_idx)
                    probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
                    return probability
    
    def _ensemble_predict(self, feature, start_padding_idx, voting_strategy='soft'):
        """执行集成预测"""
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
                    individual_results[cell_line] = {'probability': prob, 'prediction': pred}
                    
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
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
        
        return {
            'ensemble_probability': ensemble_prob,
            'ensemble_prediction': ensemble_pred,
            'num_models': len(probabilities),
            'voting_strategy': voting_strategy,
            'individual_results': individual_results
        }
    
    def predict_from_fasta(self, fasta_file, output_file=None, return_attention=False, voting_strategy='soft'):
        """
        从FASTA文件预测多个蛋白质的重要性
        
        Args:
            fasta_file: FASTA文件路径
            output_file: 输出文件路径
            return_attention: 是否返回注意力权重（仅单模型模式）
            voting_strategy: 投票策略（仅集成模式）
            
        Returns:
            results_df: 预测结果DataFrame
        """
        results = []
        
        # 读取FASTA文件
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        current_id = None
        current_seq = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                # 处理前一个序列
                if current_id is not None:
                    print(f"Predicting {current_id}...")
                    try:
                        if self.ensemble_mode:
                            result = self.predict_single_protein(current_seq, voting_strategy=voting_strategy)
                            results.append({
                                'protein_id': current_id,
                                'sequence': current_seq,
                                'sequence_length': len(current_seq),
                                'essentiality_probability': result['ensemble_probability'],
                                'PES_score': self.calculate_pes(result['ensemble_probability']),
                                'prediction': 'Essential' if result['ensemble_prediction'] else 'Non-essential',
                                'confidence': self._calculate_confidence(result['ensemble_probability']),
                                'voting_strategy': voting_strategy,
                                'num_models': result['num_models']
                            })
                        else:
                            if return_attention:
                                prob, attention = self.predict_single_protein(current_seq, return_attention=True)
                            else:
                                prob = self.predict_single_protein(current_seq)
                            
                            results.append({
                                'protein_id': current_id,
                                'sequence': current_seq,
                                'sequence_length': len(current_seq),
                                'essentiality_probability': prob,
                                'PES_score': self.calculate_pes(prob),
                                'prediction': 'Essential' if prob > 0.5 else 'Non-essential',
                                'confidence': self._calculate_confidence(prob)
                            })
                    except Exception as e:
                        print(f"Error predicting {current_id}: {e}")
                        results.append({
                            'protein_id': current_id,
                            'sequence': current_seq,
                            'sequence_length': len(current_seq),
                            'essentiality_probability': np.nan,
                            'PES_score': np.nan,
                            'prediction': 'Error',
                            'confidence': 'N/A'
                        })
                
                # 开始新序列
                current_id = line[1:]
                current_seq = ""
            else:
                current_seq += line
        
        # 处理最后一个序列
        if current_id is not None:
            print(f"Predicting {current_id}...")
            try:
                if self.ensemble_mode:
                    result = self.predict_single_protein(current_seq, voting_strategy=voting_strategy)
                    results.append({
                        'protein_id': current_id,
                        'sequence': current_seq,
                        'sequence_length': len(current_seq),
                        'essentiality_probability': result['ensemble_probability'],
                        'PES_score': self.calculate_pes(result['ensemble_probability']),
                        'prediction': 'Essential' if result['ensemble_prediction'] else 'Non-essential',
                        'confidence': self._calculate_confidence(result['ensemble_probability']),
                        'voting_strategy': voting_strategy,
                        'num_models': result['num_models']
                    })
                else:
                    if return_attention:
                        prob, attention = self.predict_single_protein(current_seq, return_attention=True)
                    else:
                        prob = self.predict_single_protein(current_seq)
                    
                    results.append({
                        'protein_id': current_id,
                        'sequence': current_seq,
                        'sequence_length': len(current_seq),
                        'essentiality_probability': prob,
                        'PES_score': self.calculate_pes(prob),
                        'prediction': 'Essential' if prob > 0.5 else 'Non-essential',
                        'confidence': self._calculate_confidence(prob)
                    })
            except Exception as e:
                print(f"Error predicting {current_id}: {e}")
                results.append({
                    'protein_id': current_id,
                    'sequence': current_seq,
                    'sequence_length': len(current_seq),
                    'essentiality_probability': np.nan,
                    'PES_score': np.nan,
                    'prediction': 'Error',
                    'confidence': 'N/A'
                })
        
        # 创建DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df
    
    def _calculate_confidence(self, probability):
        """计算置信度"""
        return 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
    
    def calculate_pes(self, probability):
        """
        计算蛋白质重要性分数 (PES)。
        根据论文定义，PES 直接等于模型 sigmoid 输出的概率值 (0~1)。

        Args:
            probability: 重要性概率 (0~1)

        Returns:
            float: PES 分数，与输入概率相同。
        """
        return float(probability)
    
    def analyze_biomarkers(self, results_df, pes_threshold=0.9):
        """
        分析潜在的生物标志物
        
        Args:
            results_df: 预测结果DataFrame
            pes_threshold: PES分数阈值
            
        Returns:
            biomarkers_df: 潜在生物标志物DataFrame
        """
        # 筛选高PES分数的蛋白质作为潜在生物标志物
        biomarkers = results_df[
            (results_df['PES_score'] >= pes_threshold) & 
            (results_df['confidence'] == 'High')
        ].copy()
        
        # 按PES分数排序
        biomarkers = biomarkers.sort_values('PES_score', ascending=False)
        
        # 添加生物标志物等级
        biomarkers['biomarker_grade'] = pd.cut(
            biomarkers['PES_score'],
            bins=[0, 0.6, 0.8, 0.95, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return biomarkers


def main():
    parser = argparse.ArgumentParser(description='Predict protein essentiality using trained PIC model(s)')
    
    # 基本参数
    parser.add_argument('--input_fasta', type=str, required=True,
                       help='Input FASTA file with protein sequences')
    parser.add_argument('--output_file', type=str, 
                       help='Output CSV file for results')
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='Device to use (cuda:7, cpu, etc.)')
    parser.add_argument('--esm_model_path', type=str, 
                       default='./pretrained_model/esm2_t33_650M_UR50D.pt',
                       help='Path to ESM2 pretrained model')
    
    # 模型选择
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_path', type=str,
                           help='Path to single trained PIC model (.pth file)')
    model_group.add_argument('--ensemble_mode', action='store_true',
                           help='Use ensemble of immune cell line models')
    
    # 集成模式参数
    parser.add_argument('--model_dir', type=str, default='result/model_train_results',
                       help='Directory containing trained models (for ensemble mode)')
    parser.add_argument('--cell_lines', type=str, nargs='+',
                       help='Specific cell lines for ensemble (default: 14 immune cell lines)')
    parser.add_argument('--voting_strategy', type=str, default='soft', choices=['soft', 'hard'],
                       help='Voting strategy for ensemble mode')
    
    # 分析参数
    parser.add_argument('--pes_threshold', type=float, default=0.9,
                       help='PES (probability) threshold for biomarker analysis')
    parser.add_argument('--biomarker_analysis', action='store_true',
                       help='Perform biomarker analysis')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_fasta):
        print(f"Error: Input FASTA file {args.input_fasta} not found!")
        return
    
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    # 设置输出文件
    if not args.output_file:
        base_name = Path(args.input_fasta).stem
        mode_suffix = "ensemble" if args.ensemble_mode else "single"
        args.output_file = f"{base_name}_predictions_{mode_suffix}.csv"
    
    # 初始化预测器
    if args.ensemble_mode:
        print("Initializing ensemble predictor with immune cell line models...")
        predictor = ProteinPredictor(
            esm_model_path=args.esm_model_path,
            device=args.device,
            ensemble_mode=True,
            model_dir=args.model_dir,
            cell_lines=args.cell_lines
        )
    else:
        print("Initializing single model predictor...")
        predictor = ProteinPredictor(
            model_path=args.model_path,
            esm_model_path=args.esm_model_path,
            device=args.device,
            ensemble_mode=False
        )
    
    # 进行预测
    print("Starting prediction...")
    if args.ensemble_mode:
        results_df = predictor.predict_from_fasta(
            fasta_file=args.input_fasta,
            output_file=args.output_file,
            voting_strategy=args.voting_strategy
        )
    else:
        results_df = predictor.predict_from_fasta(
            fasta_file=args.input_fasta,
            output_file=args.output_file
        )
    
    # 打印统计信息
    print("\n=== Prediction Summary ===")
    print(f"Total proteins: {len(results_df)}")
    print(f"Essential proteins: {len(results_df[results_df['prediction'] == 'Essential'])}")
    print(f"Non-essential proteins: {len(results_df[results_df['prediction'] == 'Non-essential'])}")
    print(f"Average PES score: {results_df['PES_score'].mean():.3f}")
    print(f"Max PES score: {results_df['PES_score'].max():.3f}")

    if args.ensemble_mode:
        print(f"Voting strategy: {args.voting_strategy}")
        print(f"Number of models: {results_df['num_models'].iloc[0] if len(results_df) > 0 else 0}")

    # 添加"交通灯"标记列
    def traffic_light(row):
        if row.prediction == 'Essential':
            return '🟢' if row.confidence == 'High' else '🟡'
        else:
            return '🔴' if row.confidence == 'High' else '🟠'

    results_df['flag'] = results_df.apply(traffic_light, axis=1)
    results_df.to_csv(args.output_file, index=False)

    # 生物标志物分析
    if args.biomarker_analysis:
        print("\n=== Biomarker Analysis ===")
        biomarkers_df = predictor.analyze_biomarkers(results_df, args.pes_threshold)
        
        if len(biomarkers_df) > 0:
            print(f"Potential biomarkers (PES >= {args.pes_threshold}): {len(biomarkers_df)}")
            print("\nTop 10 potential biomarkers:")
            print(biomarkers_df[['protein_id', 'PES_score', 'biomarker_grade']].head(10))
            
            # 保存生物标志物结果
            biomarker_file = args.output_file.replace('.csv', '_biomarkers.csv')
            biomarkers_df.to_csv(biomarker_file, index=False)
            print(f"Biomarker results saved to {biomarker_file}")
        else:
            print(f"No potential biomarkers found with PES >= {args.pes_threshold}")
    
    print("\nPrediction completed!")


if __name__ == '__main__':
    main() 