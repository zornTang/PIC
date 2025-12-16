#!/usr/bin/env python3
"""
从Gencode数据提取1034个中性粒细胞基因对应的蛋白质序列和ID

使用方法:
1. 下载Gencode注释文件和蛋白质序列文件
2. 运行脚本提取对应的蛋白质信息

需要的Gencode文件:
- gencode.v45.annotation.gtf.gz (基因注释)
- gencode.v45.pc_translations.fa.gz (蛋白质序列)
"""

import gzip
import re
from collections import defaultdict
from pathlib import Path

def read_gene_list(gene_file):
    """读取基因列表"""
    genes = []
    with open(gene_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('#'):
                genes.append(gene)
    print(f"读取到 {len(genes)} 个基因")
    return genes

def parse_gencode_gtf(gtf_file):
    """
    解析Gencode GTF文件，建立基因名到转录本/蛋白质ID的映射
    """
    gene_to_transcripts = defaultdict(list)
    transcript_to_protein = {}

    print(f"正在解析GTF文件: {gtf_file}")

    # 判断文件是否压缩
    if gtf_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    with open_func(gtf_file, mode) as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue

            if line_num % 100000 == 0:
                print(f"已处理 {line_num} 行...")

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            attributes = fields[8]

            # 解析属性
            attr_dict = {}
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr:
                    match = re.match(r'(\w+)\s+"([^"]+)"', attr)
                    if match:
                        attr_dict[match.group(1)] = match.group(2)

            # 处理转录本信息
            if feature_type == 'transcript':
                gene_name = attr_dict.get('gene_name')
                transcript_id = attr_dict.get('transcript_id')
                transcript_type = attr_dict.get('transcript_type')

                # 只处理蛋白编码转录本
                if (gene_name and transcript_id and
                    transcript_type == 'protein_coding'):
                    gene_to_transcripts[gene_name].append(transcript_id)

            # 处理CDS信息获取蛋白质ID
            elif feature_type == 'CDS':
                transcript_id = attr_dict.get('transcript_id')
                protein_id = attr_dict.get('protein_id')

                if transcript_id and protein_id:
                    transcript_to_protein[transcript_id] = protein_id

    print(f"解析完成，找到 {len(gene_to_transcripts)} 个基因")
    return gene_to_transcripts, transcript_to_protein

def read_protein_sequences(fasta_file):
    """读取蛋白质序列文件"""
    print(f"正在读取蛋白质序列文件: {fasta_file}")

    sequences = {}
    current_id = None
    current_seq = []

    # 判断文件是否压缩
    if fasta_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    with open_func(fasta_file, mode) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存上一个序列
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)

                # 解析新的序列ID
                # 格式: >ENSP00000000233.5|ENST00000000233.10|ENSG00000004059.11|OTTHUMG00000001094.4|OTTHUMT00000003223.4|ARF5-201|ARF5|998|CDS:1-543|
                parts = line[1:].split('|')
                if len(parts) >= 7:
                    protein_id = parts[0]  # ENSP ID
                    transcript_id = parts[1]  # ENST ID
                    gene_name = parts[6]  # 基因名
                    current_id = protein_id
                    current_seq = []
            else:
                if current_id:
                    current_seq.append(line)

        # 保存最后一个序列
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)

    print(f"读取到 {len(sequences)} 个蛋白质序列")
    return sequences

def extract_proteins_for_genes(genes, gene_to_transcripts, transcript_to_protein, protein_sequences):
    """为给定的基因列表提取蛋白质信息"""

    results = []
    found_genes = 0
    total_proteins = 0

    for gene in genes:
        if gene in gene_to_transcripts:
            found_genes += 1
            transcripts = gene_to_transcripts[gene]

            for transcript_id in transcripts:
                if transcript_id in transcript_to_protein:
                    protein_id = transcript_to_protein[transcript_id]

                    # 查找蛋白质序列（去掉版本号）
                    protein_base_id = protein_id.split('.')[0]
                    sequence = None

                    # 尝试完整ID和基础ID
                    for seq_id in protein_sequences:
                        if seq_id == protein_id or seq_id.startswith(protein_base_id):
                            sequence = protein_sequences[seq_id]
                            break

                    if sequence:
                        results.append({
                            'gene_name': gene,
                            'transcript_id': transcript_id,
                            'protein_id': protein_id,
                            'sequence': sequence,
                            'length': len(sequence)
                        })
                        total_proteins += 1

    print(f"找到 {found_genes}/{len(genes)} 个基因")
    print(f"提取到 {total_proteins} 个蛋白质序列")

    return results

def save_results(results, output_dir):
    """保存结果到文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存详细信息
    info_file = output_dir / "neutrophil_proteins_info.tsv"
    with open(info_file, 'w') as f:
        f.write("Gene\tTranscript_ID\tProtein_ID\tLength\n")
        for result in results:
            f.write(f"{result['gene_name']}\t{result['transcript_id']}\t"
                   f"{result['protein_id']}\t{result['length']}\n")

    # 保存FASTA序列
    fasta_file = output_dir / "neutrophil_proteins.fasta"
    with open(fasta_file, 'w') as f:
        for result in results:
            header = f">{result['protein_id']}|{result['transcript_id']}|{result['gene_name']}|{result['length']}aa"
            f.write(f"{header}\n")
            # 将序列按80字符换行
            seq = result['sequence']
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")

    # 保存蛋白质ID列表
    id_file = output_dir / "neutrophil_protein_ids.txt"
    with open(id_file, 'w') as f:
        for result in results:
            f.write(f"{result['protein_id']}\n")

    print(f"结果已保存到 {output_dir}")
    print(f"- 详细信息: {info_file}")
    print(f"- FASTA序列: {fasta_file}")
    print(f"- 蛋白质ID列表: {id_file}")

def download_gencode_files():
    """下载Gencode文件的说明"""
    print("\n=== 需要下载Gencode文件 ===")
    print("请下载以下文件到当前目录:")
    print("1. 基因注释文件:")
    print("   wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz")
    print("\n2. 蛋白质序列文件:")
    print("   wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.pc_translations.fa.gz")
    print("\n或者使用最新版本:")
    print("   https://www.gencodegenes.org/human/")

def main():
    # 配置文件路径
    gene_file = "neutrophil_genes_list.txt"
    gtf_file = "gencode.v45.annotation.gtf.gz"
    fasta_file = "gencode.v45.pc_translations.fa.gz"
    output_dir = "neutrophil_proteins_output"

    # 检查输入文件
    if not Path(gene_file).exists():
        print(f"错误: 基因列表文件 {gene_file} 不存在")
        return

    if not Path(gtf_file).exists() or not Path(fasta_file).exists():
        download_gencode_files()
        return

    try:
        # 1. 读取基因列表
        genes = read_gene_list(gene_file)

        # 2. 解析GTF文件
        gene_to_transcripts, transcript_to_protein = parse_gencode_gtf(gtf_file)

        # 3. 读取蛋白质序列
        protein_sequences = read_protein_sequences(fasta_file)

        # 4. 提取目标基因的蛋白质
        results = extract_proteins_for_genes(genes, gene_to_transcripts,
                                           transcript_to_protein, protein_sequences)

        # 5. 保存结果
        if results:
            save_results(results, output_dir)

            # 统计信息
            unique_genes = len(set(r['gene_name'] for r in results))
            print(f"\n=== 统计信息 ===")
            print(f"输入基因数: {len(genes)}")
            print(f"找到的基因数: {unique_genes}")
            print(f"提取的蛋白质数: {len(results)}")
            print(f"平均每个基因的蛋白质数: {len(results)/unique_genes:.2f}")
        else:
            print("未找到任何蛋白质序列")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()