#!/usr/bin/env python3
"""
从UniProt获取蛋白质功能注释信息

这个脚本会：
1. 使用基因名从UniProt查询对应的蛋白质信息
2. 获取详细的功能注释、GO注释、疾病相关性等
3. 整合所有信息并保存

使用UniProt REST API进行查询
"""

import requests
import time
import json
from pathlib import Path
import pandas as pd
from urllib.parse import quote
import sys

class UniProtAnnotator:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
        self.session = requests.Session()
        self.delay = 0.1  # 请求间隔，避免过于频繁

    def search_protein_by_gene(self, gene_name, organism="9606"):
        """
        通过基因名搜索UniProt蛋白质
        organism="9606" 表示人类
        """
        query = f"gene:{gene_name} AND organism_id:{organism}"
        url = f"{self.base_url}/uniprotkb/search"

        params = {
            'query': query,
            'format': 'json',
            'fields': 'accession,id,gene_names,protein_name,organism_name,length,sequence,ft_domain,ft_region,go_p,go_f,go_c,cc_function,cc_pathway,cc_disease,cc_subcellular_location,cc_interaction,cc_tissue_specificity,keyword,xref_ensembl,xref_string',
            'size': 500  # 每个基因可能有多个蛋白质
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            time.sleep(self.delay)  # 避免请求过于频繁

            data = response.json()
            return data.get('results', [])

        except requests.exceptions.RequestException as e:
            print(f"查询基因 {gene_name} 时出错: {e}")
            return []

    def get_protein_details(self, accession):
        """
        获取特定UniProt蛋白质的详细信息
        """
        url = f"{self.base_url}/uniprotkb/{accession}"

        params = {
            'format': 'json'
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            time.sleep(self.delay)

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"获取蛋白质 {accession} 详细信息时出错: {e}")
            return None

    def extract_protein_info(self, protein_data):
        """
        从UniProt数据中提取关键信息
        """
        if not protein_data:
            return None

        info = {
            'uniprot_id': protein_data.get('primaryAccession', ''),
            'entry_name': protein_data.get('uniProtkbId', ''),
            'protein_name': '',
            'gene_names': [],
            'organism': '',
            'length': 0,
            'function_description': '',
            'go_biological_process': [],
            'go_molecular_function': [],
            'go_cellular_component': [],
            'subcellular_location': '',
            'tissue_specificity': '',
            'disease_involvement': '',
            'pathways': [],
            'keywords': [],
            'domains': [],
            'ensembl_ids': [],
            'string_id': ''
        }

        # 蛋白质名称
        if 'proteinDescription' in protein_data:
            rec_name = protein_data['proteinDescription'].get('recommendedName', {})
            if 'fullName' in rec_name:
                info['protein_name'] = rec_name['fullName'].get('value', '')

        # 基因名
        if 'genes' in protein_data:
            for gene in protein_data['genes']:
                if 'geneName' in gene:
                    info['gene_names'].append(gene['geneName'].get('value', ''))

        # 生物体
        if 'organism' in protein_data:
            info['organism'] = protein_data['organism'].get('scientificName', '')

        # 序列长度
        if 'sequence' in protein_data:
            info['length'] = protein_data['sequence'].get('length', 0)

        # 功能描述
        if 'comments' in protein_data:
            for comment in protein_data['comments']:
                comment_type = comment.get('commentType', '')

                if comment_type == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        info['function_description'] = texts[0].get('value', '')

                elif comment_type == 'SUBCELLULAR LOCATION':
                    locations = comment.get('subcellularLocations', [])
                    if locations:
                        loc_names = []
                        for loc in locations:
                            if 'location' in loc:
                                loc_names.append(loc['location'].get('value', ''))
                        info['subcellular_location'] = '; '.join(loc_names)

                elif comment_type == 'TISSUE SPECIFICITY':
                    texts = comment.get('texts', [])
                    if texts:
                        info['tissue_specificity'] = texts[0].get('value', '')

                elif comment_type == 'DISEASE':
                    disease = comment.get('disease', {})
                    if 'diseaseId' in disease:
                        info['disease_involvement'] = disease.get('description', '')

                elif comment_type == 'PATHWAY':
                    texts = comment.get('texts', [])
                    if texts:
                        info['pathways'].append(texts[0].get('value', ''))

        # GO注释
        if 'geneOntologies' in protein_data:
            for go in protein_data['geneOntologies']:
                go_aspect = go.get('aspect', '')
                go_term = f"{go.get('term', '')} ({go.get('id', '')})"

                if go_aspect == 'P':  # Biological Process
                    info['go_biological_process'].append(go_term)
                elif go_aspect == 'F':  # Molecular Function
                    info['go_molecular_function'].append(go_term)
                elif go_aspect == 'C':  # Cellular Component
                    info['go_cellular_component'].append(go_term)

        # 关键词
        if 'keywords' in protein_data:
            for keyword in protein_data['keywords']:
                info['keywords'].append(keyword.get('name', ''))

        # 结构域信息
        if 'features' in protein_data:
            for feature in protein_data['features']:
                if feature.get('type') == 'Domain':
                    domain_desc = feature.get('description', '')
                    if domain_desc:
                        info['domains'].append(domain_desc)

        # 外部数据库引用
        if 'uniProtKBCrossReferences' in protein_data:
            for xref in protein_data['uniProtKBCrossReferences']:
                db_name = xref.get('database', '')
                if db_name == 'Ensembl':
                    info['ensembl_ids'].append(xref.get('id', ''))
                elif db_name == 'STRING':
                    info['string_id'] = xref.get('id', '')

        return info

def read_gene_list(gene_file):
    """读取基因列表"""
    genes = []
    with open(gene_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('#'):
                genes.append(gene)
    return genes

def main():
    gene_file = "neutrophil_genes_list.txt"  # 完整的1034个基因
    output_dir = Path("uniprot_annotations")
    output_dir.mkdir(exist_ok=True)

    # 检查基因列表文件
    if not Path(gene_file).exists():
        print(f"错误: 基因列表文件 {gene_file} 不存在")
        sys.exit(1)

    # 读取基因列表
    genes = read_gene_list(gene_file)
    print(f"读取到 {len(genes)} 个基因")

    # 初始化UniProt查询器
    annotator = UniProtAnnotator()

    # 存储所有结果
    all_proteins = []
    gene_summary = []

    print("开始查询UniProt数据库...")

    for i, gene in enumerate(genes, 1):
        print(f"正在查询 [{i}/{len(genes)}] {gene}")

        # 查询该基因的蛋白质
        proteins = annotator.search_protein_by_gene(gene)

        if proteins:
            print(f"  找到 {len(proteins)} 个蛋白质")

            for protein in proteins:
                # 提取蛋白质信息
                protein_info = annotator.extract_protein_info(protein)
                if protein_info:
                    protein_info['source_gene'] = gene
                    all_proteins.append(protein_info)

            # 基因级别的汇总
            primary_protein = proteins[0] if proteins else None
            if primary_protein:
                primary_info = annotator.extract_protein_info(primary_protein)
                if primary_info:
                    gene_summary.append({
                        'gene_name': gene,
                        'primary_uniprot_id': primary_info['uniprot_id'],
                        'protein_name': primary_info['protein_name'],
                        'function_description': primary_info['function_description'],
                        'total_variants': len(proteins)
                    })
        else:
            print(f"  未找到蛋白质")
            gene_summary.append({
                'gene_name': gene,
                'primary_uniprot_id': 'N/A',
                'protein_name': 'N/A',
                'function_description': 'N/A',
                'total_variants': 0
            })

        # 每查询50个基因保存一次（防止数据丢失）
        if i % 50 == 0:
            print(f"已完成 {i} 个基因，正在保存中间结果...")
            save_intermediate_results(all_proteins, gene_summary, output_dir, i)

    print(f"\n查询完成！共找到 {len(all_proteins)} 个蛋白质记录")

    # 保存最终结果
    save_final_results(all_proteins, gene_summary, output_dir)

def save_intermediate_results(proteins, gene_summary, output_dir, batch_num):
    """保存中间结果"""
    # 保存为JSON（完整数据）
    json_file = output_dir / f"batch_{batch_num}_proteins.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(proteins, f, indent=2, ensure_ascii=False)

def save_final_results(all_proteins, gene_summary, output_dir):
    """保存最终结果"""
    print("正在保存结果...")

    # 1. 保存详细的蛋白质信息 (JSON)
    json_file = output_dir / "neutrophil_uniprot_detailed.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_proteins, f, indent=2, ensure_ascii=False)

    # 2. 保存表格格式的主要信息
    if all_proteins:
        df_proteins = []
        for protein in all_proteins:
            df_proteins.append({
                'Gene': protein['source_gene'],
                'UniProt_ID': protein['uniprot_id'],
                'Entry_Name': protein['entry_name'],
                'Protein_Name': protein['protein_name'],
                'Length': protein['length'],
                'Function': protein['function_description'][:500] + '...' if len(protein['function_description']) > 500 else protein['function_description'],
                'Subcellular_Location': protein['subcellular_location'],
                'GO_Biological_Process_Count': len(protein['go_biological_process']),
                'GO_Molecular_Function_Count': len(protein['go_molecular_function']),
                'Keywords_Count': len(protein['keywords']),
                'Domains_Count': len(protein['domains'])
            })

        df = pd.DataFrame(df_proteins)
        tsv_file = output_dir / "neutrophil_uniprot_summary.tsv"
        df.to_csv(tsv_file, sep='\t', index=False)

    # 3. 保存基因级别汇总
    if gene_summary:
        df_genes = pd.DataFrame(gene_summary)
        gene_file = output_dir / "neutrophil_genes_uniprot_summary.tsv"
        df_genes.to_csv(gene_file, sep='\t', index=False)

    # 4. 保存GO注释汇总
    save_go_summary(all_proteins, output_dir)

    # 5. 保存UniProt ID列表
    uniprot_ids = [p['uniprot_id'] for p in all_proteins if p['uniprot_id']]
    id_file = output_dir / "neutrophil_uniprot_ids.txt"
    with open(id_file, 'w') as f:
        for uid in uniprot_ids:
            f.write(f"{uid}\n")

    print(f"结果已保存到 {output_dir}/")
    print(f"- 详细JSON数据: neutrophil_uniprot_detailed.json")
    print(f"- 蛋白质汇总表: neutrophil_uniprot_summary.tsv")
    print(f"- 基因汇总表: neutrophil_genes_uniprot_summary.tsv")
    print(f"- GO注释汇总: go_annotations_summary.tsv")
    print(f"- UniProt ID列表: neutrophil_uniprot_ids.txt")

def save_go_summary(all_proteins, output_dir):
    """保存GO注释的详细汇总"""
    go_data = []

    for protein in all_proteins:
        uniprot_id = protein['uniprot_id']
        gene = protein['source_gene']

        # 生物学过程
        for go_term in protein['go_biological_process']:
            go_data.append({
                'Gene': gene,
                'UniProt_ID': uniprot_id,
                'GO_Aspect': 'Biological Process',
                'GO_Term': go_term
            })

        # 分子功能
        for go_term in protein['go_molecular_function']:
            go_data.append({
                'Gene': gene,
                'UniProt_ID': uniprot_id,
                'GO_Aspect': 'Molecular Function',
                'GO_Term': go_term
            })

        # 细胞组分
        for go_term in protein['go_cellular_component']:
            go_data.append({
                'Gene': gene,
                'UniProt_ID': uniprot_id,
                'GO_Aspect': 'Cellular Component',
                'GO_Term': go_term
            })

    if go_data:
        df_go = pd.DataFrame(go_data)
        go_file = output_dir / "go_annotations_summary.tsv"
        df_go.to_csv(go_file, sep='\t', index=False)

if __name__ == "__main__":
    main()