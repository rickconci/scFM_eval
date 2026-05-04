input: 

[B, S, G]: batches of sets of cells with G genes each

1) How many genes? -> ['shared genes'] => ~8300
- global integer gene indices from a larger 90k set of possible genes 



2) how do we identify the shared genes?

```
# Read the global gene list
    global_df = pd.read_parquet(global_gene_list_path)
    if mode not in ['feature_id', 'feature_name']:
        raise ValueError("mode must be either 'feature_id' or 'feature_name'")
    gene_to_idx = global_df.drop_duplicates(subset=[mode]).set_index(mode).index
    
    # Create a mapping from gene to its first index
    gene_to_idx_map = {}
    for gene, idx in zip(global_df[mode], global_df.index):
        if gene not in gene_to_idx_map:
            gene_to_idx_map[gene] = idx
    # Map genes to indices
    mapped_indices = []
    mapped_count = 0
    for gene in genes:
        if gene in gene_to_idx_map:
            mapped_indices.append(gene_to_idx_map[gene])
            mapped_count += 1
        else:
            mapped_indices.append(-1)
    ...
    return mapped_indices
```



3) What normalization space? -> raw counts, log1p, log1pcpm, smth else
- raw counts



4) What if genes not present?
- Only mapped input genes get nonzeros
```
if issparse(adata_subset.X):
    X_subset = adata_subset.X.tocsr()
    # Get nonzero entries
    rows, cols = X_subset.nonzero()
    # Map column indices to global gene positions
    global_cols = gene_ids[cols]
    # Create new sparse matrix in global gene space
    adata_full.X = coo_matrix(
        (X_subset.data, (rows, global_cols)),
        shape=(adata_subset.n_obs, glist.shape[0])
    ).tocsr()
```


5) how many sets of cells per input?

```
experiment:
  latent_dim: 1024
  hidden_dim: 512
  generator_shared_embedding_dim: 128
  encoder_hidden_dim: 2048
  set_size: 100
```


6) at inferece we can set S to be anything we want?

```
Input: x shape [B, S, G].
input_projection: nn.Linear(G, hidden) — applied to the last dim, so for each of the B×S cells independently → [B, S, hidden].
DSPPBlock / SetNorm: work on dim last, with SetNorm also pooling stats over the set and feature dims in a way that’s not a fixed S (see the “set” in SetNorm — still no weight matrix of size S).
“Distribution” embedding: enc_mean = x.mean(dim=1) over S — any S works; it’s a mean pool, not a fixed 100-width layer.

B=1, S=1 – one cell, one “set” of size 1; the mean is trivial.
B=1, S=2000 – 2000 cells, one set; the encoder outputs one set-level vector after the mean (the path used in training with S≈100).
B=K, S=1 – K per-cell rows (each a trivial set of one cell); you still get one vector per set after the mean, i.e. K vectors — often what you want by treating each cell as its own set.

```



architecture

loss
1) trained end to end with generator


---


Set A -> encoder -> emb A
Set B -> encoder -> emb B

emb A -> diffusion model -> emb B 





