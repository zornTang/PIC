#!/usr/bin/env python3
"""
整合GENCODE高质量序列和UniProt功能注释

这个脚本会：
1. 读取GENCODE提取的蛋白质序列数据
2. 读取UniProt的功能注释数据
3. 通过基因名和蛋白质长度进行匹配
4. 生成整合的数据文件

输出文件包含：
- GENCODE的高质量蛋白质序列
- UniProt的丰富功能注释
- 最佳匹配的UniProt条目
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import re

def load_gencode_data(gencode_info_file, gencode_fasta_file):
    """加载GENCODE数据"""
    print("正在加载GENCODE数据...")

    # 读取GENCODE蛋白质信息
    gencode_df = pd.read_csv(gencode_info_file, sep='\t')
    print(f"加载了 {len(gencode_df)} 个GENCODE蛋白质条目")

    # 读取FASTA序列
    sequences = {}
    current_id = None
    current_seq = []

    with open(gencode_fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存上一个序列
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)

                # 解析新的序列ID
                # 格式: >ENSP00000220616.4|ENST00000220616.9|TG|2768aa
                parts = line[1:].split('|')
                if len(parts) >= 1:
                    current_id = parts[0]  # ENSP ID
                    current_seq = []
            else:
                if current_id:
                    current_seq.append(line)

        # 保存最后一个序列
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)

    print(f"加载了 {len(sequences)} 个GENCODE蛋白质序列")
    return gencode_df, sequences

def load_uniprot_data(uniprot_summary_file, uniprot_detailed_file):
    """加载UniProt数据"""
    print("正在加载UniProt数据...")

    # 读取UniProt摘要数据
    uniprot_df = pd.read_csv(uniprot_summary_file, sep='\t')
    print(f"加载了 {len(uniprot_df)} 个UniProt条目")

    # 读取详细注释数据
    detailed_data = {}
    if Path(uniprot_detailed_file).exists():
        with open(uniprot_detailed_file, 'r') as f:
            detailed_list = json.load(f)
            # 转换为字典，以uniprot_id为键
            for item in detailed_list:
                uniprot_id = item.get('uniprot_id')
                if uniprot_id:
                    detailed_data[uniprot_id] = item
        print(f"加载了 {len(detailed_data)} 个详细注释")

    return uniprot_df, detailed_data

def find_best_uniprot_match(gene_name, gencode_length, uniprot_df):
    """为GENCODE蛋白质找到最佳的UniProt匹配"""

    # 筛选同一个基因的UniProt条目
    gene_entries = uniprot_df[uniprot_df['Gene'] == gene_name].copy()

    if len(gene_entries) == 0:
        return None

    # 计算长度差异
    gene_entries['length_diff'] = abs(gene_entries['Length'] - gencode_length)

    # 优先级排序：
    # 1. 长度最相近的
    # 2. 有功能注释的（Function不为空）
    # 3. 主要UniProt ID（不以数字开头的P, Q, O开头的ID）

    # 标记主要ID
    gene_entries['is_main_id'] = gene_entries['UniProt_ID'].apply(
        lambda x: x.startswith(('P', 'Q', 'O')) and not x[1:].startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    )

    # 标记有功能注释的
    gene_entries['has_function'] = gene_entries['Function'].notna() & (gene_entries['Function'] != '')

    # 排序：优先主要ID，有功能注释，长度相近
    gene_entries = gene_entries.sort_values([
        'is_main_id',
        'has_function',
        'length_diff'
    ], ascending=[False, False, True])

    return gene_entries.iloc[0]

def integrate_data(gencode_df, gencode_sequences, uniprot_df, uniprot_detailed):
    """整合GENCODE和UniProt数据"""
    print("正在整合数据...")

    integrated_results = []
    match_stats = {'matched': 0, 'no_match': 0, 'multiple_matches': 0}

    for idx, gencode_row in gencode_df.iterrows():
        if idx % 500 == 0:
            print(f"已处理 {idx}/{len(gencode_df)} 条记录...")

        gene_name = gencode_row['Gene']
        protein_id = gencode_row['Protein_ID']
        transcript_id = gencode_row['Transcript_ID']
        gencode_length = gencode_row['Length']

        # 获取序列
        sequence = gencode_sequences.get(protein_id, '')

        # 查找最佳UniProt匹配
        best_match = find_best_uniprot_match(gene_name, gencode_length, uniprot_df)

        if best_match is not None:
            match_stats['matched'] += 1

            # 获取详细注释
            uniprot_id = best_match['UniProt_ID']
            detailed_info = uniprot_detailed.get(uniprot_id, {})

            # 组合结果
            result = {
                # GENCODE信息
                'gene_name': gene_name,
                'gencode_protein_id': protein_id,
                'gencode_transcript_id': transcript_id,
                'gencode_length': gencode_length,
                'protein_sequence': sequence,

                # UniProt基本信息
                'uniprot_id': uniprot_id,
                'uniprot_entry_name': best_match['Entry_Name'],
                'uniprot_protein_name': best_match['Protein_Name'],
                'uniprot_length': best_match['Length'],
                'length_difference': abs(gencode_length - best_match['Length']),

                # UniProt功能注释
                'function': best_match['Function'] if pd.notna(best_match['Function']) else '',
                'subcellular_location': best_match['Subcellular_Location'] if pd.notna(best_match['Subcellular_Location']) else '',
                'go_bp_count': best_match['GO_Biological_Process_Count'],
                'go_mf_count': best_match['GO_Molecular_Function_Count'],
                'keywords_count': best_match['Keywords_Count'],
                'domains_count': best_match['Domains_Count'],

                # 详细注释（如果有的话）
                'go_bp_terms': detailed_info.get('go_biological_process', []),
                'go_mf_terms': detailed_info.get('go_molecular_function', []),
                'go_cc_terms': detailed_info.get('go_cellular_component', []),
                'keywords': detailed_info.get('keywords', []),
                'domains': detailed_info.get('domains', []),
                'pathways': detailed_info.get('pathways', []),
                'disease_involvement': detailed_info.get('disease_involvement', ''),
                'tissue_specificity': detailed_info.get('tissue_specificity', '')
            }

        else:
            match_stats['no_match'] += 1
            result = {
                # GENCODE信息
                'gene_name': gene_name,
                'gencode_protein_id': protein_id,
                'gencode_transcript_id': transcript_id,
                'gencode_length': gencode_length,
                'protein_sequence': sequence,

                # 空的UniProt信息
                'uniprot_id': '',
                'uniprot_entry_name': '',
                'uniprot_protein_name': '',
                'uniprot_length': 0,
                'length_difference': 0,
                'function': '',
                'subcellular_location': '',
                'go_bp_count': 0,
                'go_mf_count': 0,
                'keywords_count': 0,
                'domains_count': 0,
                'go_bp_terms': [],
                'go_mf_terms': [],
                'go_cc_terms': [],
                'keywords': [],
                'domains': [],
                'pathways': [],
                'disease_involvement': '',
                'tissue_specificity': ''
            }

        integrated_results.append(result)

    print(f"\n=== 匹配统计 ===")
    print(f"成功匹配: {match_stats['matched']}")
    print(f"未找到匹配: {match_stats['no_match']}")
    print(f"匹配率: {match_stats['matched']/len(gencode_df)*100:.1f}%")

    return integrated_results

def save_integrated_results(results, output_dir):
    """保存整合结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"正在保存结果到 {output_dir}...")

    # 1. 保存详细的TSV文件
    tsv_file = output_dir / "integrated_gencode_uniprot.tsv"
    df = pd.DataFrame(results)

    # 选择主要列保存到TSV
    main_columns = [
        'gene_name', 'gencode_protein_id', 'gencode_transcript_id', 'gencode_length',
        'uniprot_id', 'uniprot_entry_name', 'uniprot_protein_name', 'uniprot_length', 'length_difference',
        'function', 'subcellular_location', 'go_bp_count', 'go_mf_count', 'keywords_count', 'domains_count'
    ]

    df[main_columns].to_csv(tsv_file, sep='\t', index=False)
    print(f"保存TSV文件: {tsv_file}")

    # 2. 保存完整的JSON文件（包含所有注释）
    json_file = output_dir / "integrated_gencode_uniprot_detailed.json"

    # 转换numpy类型为Python原生类型
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    converted_results = convert_types(results)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    print(f"保存JSON文件: {json_file}")

    # 3. 保存高质量FASTA文件（带UniProt注释）
    fasta_file = output_dir / "integrated_proteins.fasta"
    with open(fasta_file, 'w') as f:
        for result in results:
            if result['protein_sequence']:
                # 构建丰富的FASTA头
                header_parts = [
                    f"GENCODE:{result['gencode_protein_id']}",
                    f"Gene:{result['gene_name']}",
                    f"Length:{result['gencode_length']}aa"
                ]

                if result['uniprot_id']:
                    header_parts.extend([
                        f"UniProt:{result['uniprot_id']}",
                        f"Function:{result['function'][:100]}..." if len(result['function']) > 100 else f"Function:{result['function']}"
                    ])

                header = f">{' | '.join(header_parts)}"
                f.write(f"{header}\n")

                # 写入序列（每行80字符）
                seq = result['protein_sequence']
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")

    print(f"保存FASTA文件: {fasta_file}")

    # 4. 保存统计摘要
    summary_file = output_dir / "integration_summary.txt"
    with open(summary_file, 'w') as f:
        matched_count = sum(1 for r in results if r['uniprot_id'])
        total_count = len(results)

        f.write("=== GENCODE-UniProt整合摘要 ===\n\n")
        f.write(f"总蛋白质数: {total_count}\n")
        f.write(f"成功匹配UniProt: {matched_count}\n")
        f.write(f"匹配率: {matched_count/total_count*100:.1f}%\n\n")

        # 按基因统计
        gene_stats = defaultdict(int)
        for result in results:
            gene_stats[result['gene_name']] += 1

        f.write(f"涉及基因数: {len(gene_stats)}\n")
        f.write(f"平均每个基因的蛋白质数: {total_count/len(gene_stats):.1f}\n\n")

        # 功能注释统计
        with_function = sum(1 for r in results if r['function'])
        with_location = sum(1 for r in results if r['subcellular_location'])
        with_go = sum(1 for r in results if r['go_bp_count'] > 0 or r['go_mf_count'] > 0)

        f.write("=== 功能注释覆盖率 ===\n")
        f.write(f"有功能描述: {with_function} ({with_function/total_count*100:.1f}%)\n")
        f.write(f"有亚细胞定位: {with_location} ({with_location/total_count*100:.1f}%)\n")
        f.write(f"有GO注释: {with_go} ({with_go/total_count*100:.1f}%)\n")

    print(f"保存统计摘要: {summary_file}")

    return {
        'tsv_file': tsv_file,
        'json_file': json_file,
        'fasta_file': fasta_file,
        'summary_file': summary_file
    }

def main():
    """主函数"""
    # 配置文件路径
    gencode_info_file = "neutrophil_proteins_output/neutrophil_proteins_info.tsv"
    gencode_fasta_file = "neutrophil_proteins_output/neutrophil_proteins.fasta"
    uniprot_summary_file = "uniprot_annotations/neutrophil_uniprot_summary.tsv"
    uniprot_detailed_file = "uniprot_annotations/neutrophil_uniprot_detailed.json"
    output_dir = "integrated_proteins_output"

    # 检查输入文件
    required_files = [gencode_info_file, gencode_fasta_file, uniprot_summary_file]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"错误: 输入文件 {file_path} 不存在")
            return

    try:
        # 1. 加载数据
        gencode_df, gencode_sequences = load_gencode_data(gencode_info_file, gencode_fasta_file)
        uniprot_df, uniprot_detailed = load_uniprot_data(uniprot_summary_file, uniprot_detailed_file)

        # 2. 整合数据
        integrated_results = integrate_data(gencode_df, gencode_sequences, uniprot_df, uniprot_detailed)

        # 3. 保存结果
        output_files = save_integrated_results(integrated_results, output_dir)

        print(f"\n=== 整合完成 ===")
        print(f"整合了 {len(integrated_results)} 个蛋白质条目")
        print(f"结果文件保存在: {output_dir}")
        for file_type, file_path in output_files.items():
            print(f"- {file_type}: {file_path}")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()