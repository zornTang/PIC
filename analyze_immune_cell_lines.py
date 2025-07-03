#!/usr/bin/env python3
"""
分析PIC项目中的免疫细胞相关细胞系
"""

import pandas as pd
import os

def analyze_immune_cell_lines():
    """分析免疫细胞相关的细胞系"""
    
    print("="*80)
    print("PIC 项目中的免疫细胞相关细胞系分析")
    print("="*80)
    
    # 读取细胞系元信息
    meta_df = pd.read_csv('data/cell_line_meta_info.csv')
    
    # 筛选免疫细胞相关的细胞系
    immune_keywords = ['Haematopoietic', 'Lymphoid', 'Leukemia', 'Lymphoma', 'Myeloma']
    
    immune_cells = meta_df[
        meta_df['tissue'].str.contains('|'.join(immune_keywords), case=False, na=False) |
        meta_df['cancer_type'].str.contains('|'.join(immune_keywords), case=False, na=False) |
        meta_df['cancer_type_detail'].str.contains('|'.join(immune_keywords), case=False, na=False)
    ]
    
    print(f"总细胞系数量: {len(meta_df)}")
    print(f"免疫细胞系数量: {len(immune_cells)}")
    print(f"免疫细胞系占比: {len(immune_cells)/len(meta_df)*100:.1f}%")
    print()
    
    # 按癌症类型分组
    print("免疫细胞系按癌症类型分组:")
    print("-" * 50)
    cancer_types = immune_cells['cancer_type'].value_counts()
    for cancer_type, count in cancer_types.items():
        print(f"{cancer_type}: {count} 个细胞系")
    print()
    
    # 详细列表
    print("免疫细胞系详细列表:")
    print("-" * 80)
    print(f"{'细胞系名称':<15} {'组织':<20} {'癌症类型':<25} {'详细类型'}")
    print("-" * 80)
    
    immune_cell_names = []
    for _, row in immune_cells.iterrows():
        cell_line = row['cell_line']
        tissue = row['tissue']
        cancer_type = row['cancer_type']
        detail = row['cancer_type_detail']
        
        print(f"{cell_line:<15} {tissue:<20} {cancer_type:<25} {detail}")
        immune_cell_names.append(cell_line)
    
    print()
    print("="*80)
    
    # 分类分析
    print("免疫细胞系分类分析:")
    print("-" * 50)
    
    # 1. 急性髓系白血病
    aml_cells = immune_cells[immune_cells['cancer_type_detail'].str.contains('Acute Myeloid Leukemia', na=False)]
    print(f"急性髓系白血病 (AML): {len(aml_cells)} 个")
    for _, row in aml_cells.iterrows():
        print(f"  - {row['cell_line']}")
    print()
    
    # 2. 浆细胞骨髓瘤
    myeloma_cells = immune_cells[immune_cells['cancer_type_detail'].str.contains('Plasma Cell Myeloma', na=False)]
    print(f"浆细胞骨髓瘤 (Myeloma): {len(myeloma_cells)} 个")
    for _, row in myeloma_cells.iterrows():
        print(f"  - {row['cell_line']}")
    print()
    
    # 3. B细胞淋巴瘤
    lymphoma_cells = immune_cells[immune_cells['cancer_type_detail'].str.contains('B-Cell Non-Hodgkin', na=False)]
    print(f"B细胞非霍奇金淋巴瘤: {len(lymphoma_cells)} 个")
    for _, row in lymphoma_cells.iterrows():
        print(f"  - {row['cell_line']}")
    print()
    
    # 4. 非癌性免疫细胞
    non_cancer_cells = immune_cells[immune_cells['cancer_type_detail'].str.contains('Non-Cancerous', na=False)]
    print(f"非癌性免疫细胞: {len(non_cancer_cells)} 个")
    for _, row in non_cancer_cells.iterrows():
        print(f"  - {row['cell_line']}")
    print()
    
    print("="*80)
    
    # 生成训练命令
    print("训练免疫细胞系的命令:")
    print("-" * 50)
    
    immune_cell_list = ','.join(immune_cell_names)
    
    print("# 训练所有免疫细胞系:")
    print(f"python code/train_all_cell_lines.py \\")
    print(f"    --specific_cell_lines \"{immune_cell_list}\" \\")
    print(f"    --device cuda:7 \\")
    print(f"    --num_epochs 15")
    print()
    
    print("# 测试训练 (前3个免疫细胞系):")
    test_cells = ','.join(immune_cell_names[:3])
    print(f"python code/train_all_cell_lines.py \\")
    print(f"    --specific_cell_lines \"{test_cells}\" \\")
    print(f"    --device cuda:7 \\")
    print(f"    --num_epochs 5")
    print()
    
    # 按类型分别训练的命令
    print("# 按类型分别训练:")
    print()
    
    if len(aml_cells) > 0:
        aml_names = ','.join(aml_cells['cell_line'].tolist())
        print(f"# 急性髓系白血病:")
        print(f"python code/train_all_cell_lines.py --specific_cell_lines \"{aml_names}\" --device cuda:7")
        print()
    
    if len(myeloma_cells) > 0:
        myeloma_names = ','.join(myeloma_cells['cell_line'].tolist())
        print(f"# 浆细胞骨髓瘤:")
        print(f"python code/train_all_cell_lines.py --specific_cell_lines \"{myeloma_names}\" --device cuda:7")
        print()
    
    if len(lymphoma_cells) > 0:
        lymphoma_names = ','.join(lymphoma_cells['cell_line'].tolist())
        print(f"# B细胞淋巴瘤:")
        print(f"python code/train_all_cell_lines.py --specific_cell_lines \"{lymphoma_names}\" --device cuda:7")
        print()
    
    print("="*80)
    
    # 保存免疫细胞系列表
    immune_cells.to_csv('immune_cell_lines.csv', index=False)
    print(f"免疫细胞系信息已保存到: immune_cell_lines.csv")
    
    # 创建训练脚本
    with open('train_immune_cells.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 训练所有免疫细胞系的脚本\n\n")
        f.write("echo \"开始训练免疫细胞系...\"\n")
        f.write("echo \"免疫细胞系数量: {}\"\n".format(len(immune_cells)))
        f.write("\n")
        f.write("conda activate PIC\n")
        f.write("export MKL_SERVICE_FORCE_INTEL=1\n")
        f.write("\n")
        f.write("python code/train_all_cell_lines.py \\\n")
        f.write(f"    --specific_cell_lines \"{immune_cell_list}\" \\\n")
        f.write("    --device cuda:7 \\\n")
        f.write("    --num_epochs 15 \\\n")
        f.write("    --learning_rate 1e-5\n")
        f.write("\n")
        f.write("echo \"免疫细胞系训练完成！\"\n")
    
    os.chmod('train_immune_cells.sh', 0o755)
    print(f"训练脚本已创建: train_immune_cells.sh")
    print()
    
    return immune_cells

if __name__ == "__main__":
    immune_cells = analyze_immune_cell_lines() 