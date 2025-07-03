import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from module.PIC import PIC


class ProteinPredictor:
    """蛋白质重要性预测器"""
    
    def __init__(self, model_path, esm_model_path, device='cuda:0'):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的PIC模型路径
            esm_model_path: ESM2预训练模型路径
            device: 计算设备
        """
        self.device = device
        self.esm_model_path = esm_model_path
        
        # 加载ESM2模型
        print("Loading ESM2 model...")
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_path)
        self.esm_model.eval()
        if torch.cuda.is_available():
            self.esm_model = self.esm_model.to(device)
        
        # 加载训练好的PIC模型
        print("Loading PIC model...")
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
    
    def predict_single_protein(self, sequence, return_attention=False):
        """
        预测单个蛋白质的重要性
        
        Args:
            sequence: 蛋白质序列
            return_attention: 是否返回注意力权重
            
        Returns:
            prediction: 预测概率
            attention_weights: 注意力权重（可选）
        """
        # 提取特征
        feature, start_padding_idx = self.extract_sequence_embedding(sequence)
        
        # 添加batch维度
        feature = feature.unsqueeze(0)
        start_padding_idx = start_padding_idx.unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            if return_attention:
                logits, attention_weights = self.pic_model(feature, start_padding_idx, get_attention=True)
                probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
                return probability, attention_weights.cpu().numpy()
            else:
                logits = self.pic_model(feature, start_padding_idx)
                probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
                return probability
    
    def predict_from_fasta(self, fasta_file, output_file=None, return_attention=False):
        """
        从FASTA文件预测多个蛋白质的重要性
        
        Args:
            fasta_file: FASTA文件路径
            output_file: 输出文件路径
            return_attention: 是否返回注意力权重
            
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
                        if return_attention:
                            prob, attention = self.predict_single_protein(current_seq, return_attention=True)
                            results.append({
                                'protein_id': current_id,
                                'sequence': current_seq,
                                'sequence_length': len(current_seq),
                                'essentiality_probability': prob,
                                'PES_score': self.calculate_pes(prob),
                                'prediction': 'Essential' if prob > 0.5 else 'Non-essential',
                                'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'
                            })
                        else:
                            prob = self.predict_single_protein(current_seq)
                            results.append({
                                'protein_id': current_id,
                                'sequence': current_seq,
                                'sequence_length': len(current_seq),
                                'essentiality_probability': prob,
                                'PES_score': self.calculate_pes(prob),
                                'prediction': 'Essential' if prob > 0.5 else 'Non-essential',
                                'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'
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
                    'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.1 else 'Low'
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
    parser = argparse.ArgumentParser(description='Predict protein essentiality using trained PIC model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained PIC model (.pth file)')
    parser.add_argument('--esm_model_path', type=str, 
                       default='./pretrained_model/esm2_t33_650M_UR50D.pt',
                       help='Path to ESM2 pretrained model')
    parser.add_argument('--input_fasta', type=str, required=True,
                       help='Input FASTA file with protein sequences')
    parser.add_argument('--output_file', type=str, 
                       help='Output CSV file for results')
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--pes_threshold', type=float, default=0.9,
                       help='PES (probability) threshold for biomarker analysis')
    parser.add_argument('--biomarker_analysis', action='store_true',
                       help='Perform biomarker analysis')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_fasta):
        print(f"Error: Input FASTA file {args.input_fasta} not found!")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    # 设置输出文件
    if not args.output_file:
        base_name = Path(args.input_fasta).stem
        args.output_file = f"{base_name}_predictions.csv"
    
    # 初始化预测器
    predictor = ProteinPredictor(
        model_path=args.model_path,
        esm_model_path=args.esm_model_path,
        device=args.device
    )
    
    # 进行预测
    print("Starting prediction...")
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

    # 添加“交通灯”标记列，并覆盖保存带 flag 的结果文件
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