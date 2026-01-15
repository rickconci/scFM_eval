import anndata as ad

path = '/lotterlab/users/riccardo/ML_BIO/scFM_repos/scFM_eval/__output/brca_full/extract_embeddings_embeddings/data.h5ad'

adata = ad.read_h5ad(path)
print(adata)
print(adata.obsm['X_concat_scConcept_scimilarity'].shape)
print("\n" + "="*80)
print("Value counts for each column:")
print("="*80)
print("\ncellType:")
print(adata.obs['cellType'].value_counts())
print("\ncell_types:")
print(adata.obs['cell_types'].value_counts())
print("\ncelltype:")
print(adata.obs['celltype'].value_counts())

print("\n" + "="*80)
print("Comparison checks:")
print("="*80)

# Check if columns are identical (same values, same order)
print("\n1. Are columns identical (same values, same order)?")
print(f"   cellType == cell_types: {(adata.obs['cellType'] == adata.obs['cell_types']).all()}")
print(f"   cellType == celltype: {(adata.obs['cellType'] == adata.obs['celltype']).all()}")
print(f"   cell_types == celltype: {(adata.obs['cell_types'] == adata.obs['celltype']).all()}")

# Check if they have the same unique values (ignoring order)
print("\n2. Do columns have the same unique values (ignoring order)?")
print(f"   cellType == cell_types: {set(adata.obs['cellType'].unique()) == set(adata.obs['cell_types'].unique())}")
print(f"   cellType == celltype: {set(adata.obs['cellType'].unique()) == set(adata.obs['celltype'].unique())}")
print(f"   cell_types == celltype: {set(adata.obs['cell_types'].unique()) == set(adata.obs['celltype'].unique())}")

# Check if value counts are the same
print("\n3. Do columns have the same value counts?")
vc1 = adata.obs['cellType'].value_counts().sort_index()
vc2 = adata.obs['cell_types'].value_counts().sort_index()
vc3 = adata.obs['celltype'].value_counts().sort_index()
print(f"   cellType == cell_types: {vc1.equals(vc2)}")
print(f"   cellType == celltype: {vc1.equals(vc3)}")
print(f"   cell_types == celltype: {vc2.equals(vc3)}")

# Show where they differ if not identical
print("\n4. Differences (if any):")
if not (adata.obs['cellType'] == adata.obs['cell_types']).all():
    diff_mask = adata.obs['cellType'] != adata.obs['cell_types']
    print(f"   cellType vs cell_types: {diff_mask.sum()} differences")
    print(f"   First 10 differences:")
    diff_df = adata.obs.loc[diff_mask, ['cellType', 'cell_types']].head(10)
    print(diff_df.to_string())

if not (adata.obs['cellType'] == adata.obs['celltype']).all():
    diff_mask = adata.obs['cellType'] != adata.obs['celltype']
    print(f"   cellType vs celltype: {diff_mask.sum()} differences")
    print(f"   First 10 differences:")
    diff_df = adata.obs.loc[diff_mask, ['cellType', 'celltype']].head(10)
    print(diff_df.to_string())

if not (adata.obs['cell_types'] == adata.obs['celltype']).all():
    diff_mask = adata.obs['cell_types'] != adata.obs['celltype']
    print(f"   cell_types vs celltype: {diff_mask.sum()} differences")
    print(f"   First 10 differences:")
    diff_df = adata.obs.loc[diff_mask, ['cell_types', 'celltype']].head(10)
    print(diff_df.to_string())



