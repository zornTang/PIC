#!/usr/bin/env python3
"""
PIC-cell 完整训练和集成学习流程
1. 训练所有细胞系的PIC模型
2. 使用软投票策略构建集成模型
"""

import os
import sys
import subprocess
import time
import pandas as pd
import argparse
from datetime import datetime
import logging


def setup_logging(log_file):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_environment():
    """检查环境和依赖"""
    print("检查环境...")
    
    required_files = [
        'data/cell_data.pkl',
        'data/cell_line_meta_info.csv',
        'pretrained_model/esm2_t33_650M_UR50D.pt',
        'code/train_all_cell_lines.py',
        'code/ensemble_pic_cell.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"缺少必需文件: {missing_files}")
        return False
    
    return True


def train_all_cell_lines(args, logger):
    """训练所有细胞系模型"""
    logger.info("开始训练所有细胞系模型...")
    
    # 构建训练命令，传递所有main.py中的参数
    cmd = [
        'python', 'code/train_all_cell_lines.py',
        '--data_path', args.data_path,
        '--cell_line_meta_file', args.cell_line_meta_file,
        '--esm_model_path', args.esm_model_path,
        '--output_dir', args.model_output_dir,
        '--embedding_dir', args.embedding_dir,
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        # 添加main.py中的其他训练参数
        '--test_ratio', str(args.test_ratio),
        '--val_ratio', str(args.val_ratio),
        '--linear_drop', str(args.linear_drop),
        '--attn_drop', str(args.attn_drop),
        '--max_length', str(args.max_length),
        '--feature_length', str(args.feature_length),
        '--input_size', str(args.input_size),
        '--hidden_size', str(args.hidden_size),
        '--output_size', str(args.output_size),
        '--random_seed', str(args.random_seed)
    ]
    
    if args.specific_cell_lines:
        cmd.extend(['--specific_cell_lines', args.specific_cell_lines])
    
    if args.overwrite:
        cmd.append('--overwrite')
    
    # 设置环境变量
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    
    logger.info(f"训练命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"所有细胞系训练完成，耗时: {training_time:.2f}秒")
            return True
        else:
            logger.error(f"训练失败，返回码: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        return False


def build_ensemble_model(args, logger):
    """构建集成模型"""
    logger.info("开始构建PIC-cell集成模型...")
    
    # 构建集成命令
    cmd = [
        'python', 'code/ensemble_pic_cell.py',
        '--model_dir', args.model_output_dir,
        '--esm_model_path', args.esm_model_path,
        '--data_path', args.data_path,
        '--device', args.device,
        '--test_size', str(args.test_size),
        '--voting_strategy', args.voting_strategy,
        '--output_dir', args.ensemble_output_dir
    ]
    
    if args.ensemble_cell_lines:
        cmd.extend(['--cell_lines'] + args.ensemble_cell_lines.split(','))
    
    # 设置环境变量
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    
    logger.info(f"集成命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        end_time = time.time()
        
        ensemble_time = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"集成模型构建完成，耗时: {ensemble_time:.2f}秒")
            return True
        else:
            logger.error(f"集成模型构建失败，返回码: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"集成过程中发生错误: {e}")
        return False


def analyze_results(args, logger):
    """分析结果"""
    logger.info("分析训练和集成结果...")
    
    # 统计训练的模型数量
    model_count = 0
    if os.path.exists(args.model_output_dir):
        for folder in os.listdir(args.model_output_dir):
            if folder.startswith('PIC_'):
                model_path = os.path.join(args.model_output_dir, folder, f'{folder}_model.pth')
                if os.path.exists(model_path):
                    model_count += 1
    
    logger.info(f"成功训练的细胞系模型数量: {model_count}")
    
    # 检查集成结果
    ensemble_results = []
    if os.path.exists(args.ensemble_output_dir):
        for file in os.listdir(args.ensemble_output_dir):
            if file.endswith('.csv'):
                ensemble_results.append(file)
    
    logger.info(f"生成的集成结果文件: {ensemble_results}")
    
    # 读取并显示集成性能
    metrics_file = os.path.join(args.ensemble_output_dir, f'metrics_{args.voting_strategy}.csv')
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        logger.info("集成模型性能指标:")
        for _, row in metrics_df.iterrows():
            logger.info(f"  Accuracy: {row['accuracy']:.4f}")
            logger.info(f"  Precision: {row['precision']:.4f}")
            logger.info(f"  Recall: {row['recall']:.4f}")
            logger.info(f"  F1-Score: {row['f1_score']:.4f}")
            logger.info(f"  AUC-ROC: {row['auc_roc']:.4f}")
            logger.info(f"  AUC-PR: {row['auc_pr']:.4f}")
    
    return model_count, ensemble_results


def main():
    parser = argparse.ArgumentParser(description='PIC-cell 完整训练和集成学习流程')
    
    # 数据和模型路径
    parser.add_argument('--data_path', type=str, default='data/cell_data.pkl',
                       help='细胞系数据文件路径')
    parser.add_argument('--cell_line_meta_file', type=str, default='data/cell_line_meta_info.csv',
                       help='细胞系元信息文件路径')
    parser.add_argument('--esm_model_path', type=str, default='pretrained_model/esm2_t33_650M_UR50D.pt',
                       help='ESM2预训练模型路径')
    
    # 输出目录
    parser.add_argument('--model_output_dir', type=str, default='result/model_train_results',
                       help='细胞系模型保存目录')
    parser.add_argument('--embedding_dir', type=str, default='result/seq_embedding',
                       help='序列嵌入保存目录')
    parser.add_argument('--ensemble_output_dir', type=str, default='result/ensemble_results',
                       help='集成模型结果保存目录')
    
    # 训练参数 (使用main.py中的参数设置)
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='训练设备')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='学习率')
    
    # 添加main.py中的其他训练参数
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--linear_drop', type=float, default=0.1,
                       help='线性层dropout率')
    parser.add_argument('--attn_drop', type=float, default=0.3,
                       help='注意力层dropout率')
    parser.add_argument('--max_length', type=int, default=1000,
                       help='最大序列长度')
    parser.add_argument('--feature_length', type=int, default=1280,
                       help='特征向量长度')
    parser.add_argument('--input_size', type=int, default=1280,
                       help='输入特征维度')
    parser.add_argument('--hidden_size', type=int, default=320,
                       help='隐藏层维度')
    parser.add_argument('--output_size', type=int, default=1,
                       help='输出维度')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    # 细胞系选择
    parser.add_argument('--specific_cell_lines', type=str,
                       help='指定要训练的细胞系（用逗号分隔）')
    parser.add_argument('--ensemble_cell_lines', type=str,
                       help='指定用于集成的细胞系（用逗号分隔）')
    
    # 集成参数
    parser.add_argument('--voting_strategy', type=str, default='soft',
                       choices=['soft', 'hard', 'weighted', 'all'],
                       help='投票策略')
    parser.add_argument('--test_size', type=int, default=1000,
                       help='测试数据大小')
    
    # 流程控制
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练，直接进行集成')
    parser.add_argument('--skip_ensemble', action='store_true',
                       help='跳过集成，只进行训练')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的模型')
    
    args = parser.parse_args()
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，请确保所有必需文件存在")
        return
    
    # 创建输出目录
    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)
    os.makedirs(args.ensemble_output_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.ensemble_output_dir, f'pic_cell_workflow_{timestamp}.log')
    logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("PIC-cell 完整训练和集成学习流程")
    logger.info("="*80)
    logger.info(f"开始时间: {datetime.now()}")
    
    total_start_time = time.time()
    
    # 步骤1: 训练所有细胞系模型
    if not args.skip_training:
        logger.info("\n步骤1: 训练所有细胞系模型")
        logger.info("-" * 50)
        
        training_success = train_all_cell_lines(args, logger)
        if not training_success:
            logger.error("训练失败，终止流程")
            return
    else:
        logger.info("跳过训练步骤")
    
    # 步骤2: 构建集成模型
    if not args.skip_ensemble:
        logger.info("\n步骤2: 构建PIC-cell集成模型")
        logger.info("-" * 50)
        
        ensemble_success = build_ensemble_model(args, logger)
        if not ensemble_success:
            logger.error("集成模型构建失败")
            return
    else:
        logger.info("跳过集成步骤")
    
    # 步骤3: 分析结果
    logger.info("\n步骤3: 分析结果")
    logger.info("-" * 50)
    
    model_count, ensemble_results = analyze_results(args, logger)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("PIC-cell 流程完成总结")
    logger.info("="*80)
    logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    logger.info(f"训练的细胞系模型数量: {model_count}")
    logger.info(f"集成结果文件数量: {len(ensemble_results)}")
    logger.info(f"投票策略: {args.voting_strategy}")
    logger.info(f"模型保存位置: {args.model_output_dir}")
    logger.info(f"集成结果位置: {args.ensemble_output_dir}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"完成时间: {datetime.now()}")
    
    print(f"\n{'='*80}")
    print("PIC-cell 训练和集成学习流程完成！")
    print(f"{'='*80}")
    print(f"详细日志: {log_file}")
    print(f"模型目录: {args.model_output_dir}")
    print(f"集成结果: {args.ensemble_output_dir}")


if __name__ == "__main__":
    main() 