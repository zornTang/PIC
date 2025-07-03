#!/usr/bin/env python3
"""
环境检查脚本
验证PIC批量训练所需的所有文件和依赖
"""

import os
import sys
import pandas as pd
import pickle

def check_python_packages():
    """检查Python包"""
    print("检查Python包...")
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 'esm', 'argparse', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        return False
    return True

def check_files():
    """检查必需文件"""
    print("\n检查必需文件...")
    required_files = [
        'data/cell_data.pkl',
        'data/cell_line_meta_info.csv',
        'code/main.py',
        'code/embedding.py',
        'code/train_all_cell_lines.py'
    ]
    
    optional_files = [
        'pretrained_model/esm2_t33_650M_UR50D.pt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 不存在")
            missing_files.append(file_path)
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"⚠ {file_path} - 不存在 (可选，训练时需要)")
    
    if missing_files:
        print(f"\n缺少以下必需文件: {', '.join(missing_files)}")
        return False
    return True

def check_data_integrity():
    """检查数据完整性"""
    print("\n检查数据完整性...")
    
    try:
        # 检查细胞系数据
        with open('data/cell_data.pkl', 'rb') as f:
            cell_data = pickle.load(f)
        print(f"✓ 细胞系数据: {len(cell_data)} 个蛋白质")
        
        # 检查细胞系元信息
        meta_df = pd.read_csv('data/cell_line_meta_info.csv')
        print(f"✓ 细胞系元信息: {len(meta_df)} 个细胞系")
        
        # 检查数据一致性
        cell_lines_in_data = [col for col in cell_data.columns if col not in ['index', 'ID', 'sequence']]
        cell_lines_in_meta = meta_df['cell_line'].unique().tolist()
        
        print(f"✓ 数据中的细胞系: {len(cell_lines_in_data)} 个")
        print(f"✓ 元信息中的细胞系: {len(cell_lines_in_meta)} 个")
        
        # 显示前几个细胞系
        print(f"✓ 前5个细胞系: {cell_lines_in_meta[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据检查失败: {e}")
        return False

def check_directories():
    """检查和创建必要目录"""
    print("\n检查输出目录...")
    
    directories = [
        'result',
        'result/model_train_results',
        'result/seq_embedding'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 创建目录: {directory}")
        else:
            print(f"✓ 目录存在: {directory}")
    
    return True

def check_gpu():
    """检查GPU可用性"""
    print("\n检查GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            print(f"✓ GPU可用: {gpu_count} 个GPU")
            print(f"✓ 当前GPU: {gpu_name}")
            print(f"✓ 推荐使用: --device cuda:7")
        else:
            print("⚠ GPU不可用，将使用CPU训练")
            print("✓ 推荐使用: --device cpu")
        return True
    except Exception as e:
        print(f"✗ GPU检查失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("PIC 批量训练环境检查")
    print("="*60)
    
    checks = [
        check_python_packages,
        check_files,
        check_data_integrity,
        check_directories,
        check_gpu
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 环境检查通过！可以开始批量训练")
        print("\n建议的测试命令:")
        print("python code/train_all_cell_lines.py \\")
        print("    --specific_cell_lines 'A549' \\")
        print("    --num_epochs 2 \\")
        print("    --device cuda:7")
    else:
        print("✗ 环境检查失败，请解决上述问题后再运行")
        sys.exit(1)
    
    print("="*60)

if __name__ == "__main__":
    main() 