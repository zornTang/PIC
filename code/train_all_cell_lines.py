#!/usr/bin/env python3
"""
批量训练所有细胞系的PIC模型
这个脚本会自动读取细胞系列表，并为每个细胞系训练一个独立的模型
"""

import os
import sys
import pandas as pd
import argparse
import subprocess
import time
from pathlib import Path
import logging
from datetime import datetime


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


def load_cell_lines(meta_file):
    """加载细胞系列表"""
    try:
        df = pd.read_csv(meta_file)
        cell_lines = df['cell_line'].unique().tolist()
        logging.info(f"找到 {len(cell_lines)} 个细胞系")
        return cell_lines
    except Exception as e:
        logging.error(f"加载细胞系列表失败: {e}")
        return []


def check_prerequisites(args):
    """检查必要的文件和目录"""
    required_files = [
        args.data_path,
        args.cell_line_meta_file,
        args.esm_model_path,
        'code/embedding.py',
        'code/main.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"必需文件不存在: {file_path}")
            return False
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)
    
    return True


def generate_fasta_for_cell_line(cell_line, data_path, output_dir):
    """为指定细胞系生成FASTA文件"""
    fasta_file = os.path.join(output_dir, f"{cell_line}_sequences.fasta")
    
    # 调用embedding.py的generate_fasta函数
    cmd = [
        'python', 'code/embedding.py',
        '--data_path', data_path,
        '--fasta_file', fasta_file,
        '--label_name', cell_line
    ]
    
    try:
        # 只生成FASTA文件，不提取embedding
        import pickle
        dataset = pd.read_pickle(data_path)
        
        # 检查细胞系是否存在于数据中
        if cell_line not in dataset.columns:
            logging.warning(f"细胞系 {cell_line} 不存在于数据中，跳过")
            return None
            
        with open(fasta_file, 'w') as fasta:
            count = 0
            for index, row in dataset.iterrows():
                if pd.notna(row.get(cell_line)):  # 只处理有标签的数据
                    sequence = row['sequence'].replace('*', '')
                    id_str = f"{row['index']}_{row['ID']}_{int(row[cell_line])}"
                    fasta.write(f'>{id_str}\n{sequence}\n')
                    count += 1
        
        if count > 0:
            logging.info(f"为细胞系 {cell_line} 生成了 {count} 个序列的FASTA文件")
            return fasta_file
        else:
            logging.warning(f"细胞系 {cell_line} 没有有效数据，跳过")
            os.remove(fasta_file)
            return None
            
    except Exception as e:
        logging.error(f"生成 {cell_line} 的FASTA文件失败: {e}")
        return None


def extract_embeddings(cell_line, fasta_file, embedding_dir, esm_model_path, device):
    """提取序列嵌入"""
    cmd = [
        'python', 'code/embedding.py',
        '--data_path', 'data/cell_data.pkl',  # 占位符，实际不使用
        '--fasta_file', fasta_file,
        '--model_name', esm_model_path,
        '--label_name', cell_line,
        '--output_dir', embedding_dir,
        '--device', device,
        '--truncation_seq_length', '1024'
    ]
    
    try:
        logging.info(f"开始提取 {cell_line} 的序列嵌入...")
        # 设置环境变量解决MKL兼容性问题
        env = os.environ.copy()
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        
        if result.returncode == 0:
            logging.info(f"成功提取 {cell_line} 的序列嵌入")
            return True
        else:
            logging.error(f"提取 {cell_line} 嵌入失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"提取 {cell_line} 嵌入超时")
        return False
    except Exception as e:
        logging.error(f"提取 {cell_line} 嵌入时发生错误: {e}")
        return False


def train_cell_line_model(cell_line, args):
    """训练单个细胞系的模型"""
    cmd = [
        'python', 'code/main.py',
        '--data_path', args.data_path,
        '--label_name', cell_line,
        '--feature_dir', args.embedding_dir,
        '--save_path', args.output_dir,
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
        '--random_seed', str(args.random_seed),
        '--model_variant', args.model_variant,
        '--num_heads', str(args.num_heads),
        '--cnn_channels', str(args.cnn_channels),
        '--cnn_kernel_size', str(args.cnn_kernel_size),
        '--cnn_layers', str(args.cnn_layers),
        '--cnn_drop', str(args.cnn_drop)
    ]
    
    # 设置环境变量解决MKL兼容性问题
    env = os.environ.copy()
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    
    try:
        logging.info(f"开始训练 {cell_line} 模型...")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)  # 2小时超时
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            logging.info(f"成功训练 {cell_line} 模型，耗时: {training_time:.2f}秒")
            return True, training_time
        else:
            logging.error(f"训练 {cell_line} 模型失败: {result.stderr}")
            return False, training_time
            
    except subprocess.TimeoutExpired:
        logging.error(f"训练 {cell_line} 模型超时")
        return False, 0
    except Exception as e:
        logging.error(f"训练 {cell_line} 模型时发生错误: {e}")
        return False, 0


def train_all_cell_lines(args):
    """训练所有细胞系的主函数"""
    # 设置日志
    log_file = os.path.join(args.output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)
    
    # 检查前置条件
    if not check_prerequisites(args):
        logger.error("前置条件检查失败，退出")
        return
    
    # 加载细胞系列表
    cell_lines = load_cell_lines(args.cell_line_meta_file)
    if not cell_lines:
        logger.error("无法加载细胞系列表，退出")
        return
    
    # 如果指定了特定细胞系，只训练这些
    if args.specific_cell_lines:
        specific_lines = args.specific_cell_lines.split(',')
        # 去重并过滤有效的细胞系
        specified_cells = list(set([cl.strip() for cl in specific_lines if cl.strip()]))
        cell_lines = [cl for cl in specified_cells if cl in cell_lines]
        logger.info(f"只训练指定的细胞系: {cell_lines}")
    
    # 训练统计
    total_cell_lines = len(cell_lines)
    successful_trainings = 0
    failed_trainings = 0
    skipped_trainings = 0
    training_times = []
    
    logger.info(f"开始批量训练 {total_cell_lines} 个细胞系")
    
    for i, cell_line in enumerate(cell_lines, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"进度: {i}/{total_cell_lines} - 训练细胞系: {cell_line}")
        logger.info(f"{'='*60}")
        
        # 检查是否已经训练过
        model_path = os.path.join(args.output_dir, f"PIC_{cell_line}", f"PIC_{cell_line}_model.pth")
        if os.path.exists(model_path) and not args.overwrite:
            logger.info(f"模型已存在，跳过: {model_path}")
            skipped_trainings += 1
            continue
        
        try:
            # 步骤1: 生成FASTA文件
            fasta_file = generate_fasta_for_cell_line(cell_line, args.data_path, args.output_dir)
            if not fasta_file:
                logger.warning(f"跳过细胞系 {cell_line}（无有效数据）")
                skipped_trainings += 1
                continue
            
            # 步骤2: 提取序列嵌入
            if not extract_embeddings(cell_line, fasta_file, args.embedding_dir, args.esm_model_path, args.device):
                logger.error(f"跳过细胞系 {cell_line}（嵌入提取失败）")
                failed_trainings += 1
                continue
            
            # 步骤3: 训练模型
            success, training_time = train_cell_line_model(cell_line, args)
            
            if success:
                successful_trainings += 1
                training_times.append(training_time)
                logger.info(f"✓ 成功完成 {cell_line} 的训练")
            else:
                failed_trainings += 1
                logger.error(f"✗ {cell_line} 训练失败")
            
            # 清理临时文件
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
                
        except Exception as e:
            logger.error(f"处理细胞系 {cell_line} 时发生未预期错误: {e}")
            failed_trainings += 1
    
    # 训练总结
    logger.info(f"\n{'='*60}")
    logger.info("批量训练完成！")
    logger.info(f"{'='*60}")
    logger.info(f"总细胞系数: {total_cell_lines}")
    logger.info(f"成功训练: {successful_trainings}")
    logger.info(f"训练失败: {failed_trainings}")
    logger.info(f"跳过训练: {skipped_trainings}")
    
    if training_times:
        avg_time = sum(training_times) / len(training_times)
        logger.info(f"平均训练时间: {avg_time:.2f}秒")
        logger.info(f"总训练时间: {sum(training_times):.2f}秒")
    
    logger.info(f"日志文件: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='批量训练所有细胞系的PIC模型')
    
    # 必需参数
    parser.add_argument('--data_path', type=str, default='data/cell_data.pkl',
                       help='细胞系数据文件路径')
    parser.add_argument('--cell_line_meta_file', type=str, default='data/cell_line_meta_info.csv',
                       help='细胞系元信息文件路径')
    parser.add_argument('--esm_model_path', type=str, default='pretrained_model/esm2_t33_650M_UR50D.pt',
                       help='ESM2预训练模型路径')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='result/model_train_results',
                       help='模型保存目录')
    parser.add_argument('--embedding_dir', type=str, default='result/seq_embedding',
                       help='序列嵌入保存目录')
    
    # 训练参数 (与main.py保持一致)
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='训练设备 (cuda:7, cpu等)')
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
    parser.add_argument('--model_variant', type=str, default='attention',
                       choices=['attention','cnn','avgpool'],
                       help='骨干结构：注意力、CNN 或平均池化')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='注意力头数（仅 attention 模型生效）')
    parser.add_argument('--cnn_channels', type=int, default=256,
                       help='CNN 中间通道数（仅 cnn 模型生效）')
    parser.add_argument('--cnn_kernel_size', type=int, default=5,
                       help='CNN 卷积核大小')
    parser.add_argument('--cnn_layers', type=int, default=2,
                       help='CNN 堆叠层数')
    parser.add_argument('--cnn_drop', type=float, default=0.1,
                       help='CNN 层后的 dropout 比例')
    
    # 其他选项
    parser.add_argument('--specific_cell_lines', type=str,
                       help='只训练指定的细胞系（用逗号分隔）')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的模型')
    
    args = parser.parse_args()
    
    # 开始批量训练
    train_all_cell_lines(args)


if __name__ == "__main__":
    main() 
