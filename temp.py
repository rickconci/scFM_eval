import pickle 
import torch
import anndata as ad
import scanpy as sc
import os

data_path = '/orcd/data/omarabu/001/Omnicell_datasets/bio_batch_eval_data/dkd.h5ad'


# for file in os.listdir(data_path):
#     print('loading file: ', file)
#     data = ad.read_h5ad(os.path.join(data_path, file), backed='r')
#     print(data.var_names[:100])
#     for v in data.var:
#         print(data.var[v].value_counts())
#     for o in data.obs:
#         print(data.obs[o].value_counts())


gene_manager_path = '/orcd/data/omarabu/001/Omnicell_datasets/protocol_embeddings/gene_manager_serialized_2.pkl'
with open(gene_manager_path, 'rb') as f:
    gene_manager = pickle.load(f)

print(gene_manager.keys())
#print(gene_manager['total_genes'])
print('shared genes: ', len(gene_manager['shared_genes']))
print(gene_manager['shared_genes'])

import pandas as pd

global_gene_mapping_v2_path = '/orcd/data/omarabu/001/Omnicell_datasets/protocol_embeddings/global_gene_mapping_v2.parquet'
global_gene_mapping_v2 = pd.read_parquet(global_gene_mapping_v2_path)
print(global_gene_mapping_v2.head())