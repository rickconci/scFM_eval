"""
Baseline embedding methods for batch effects evaluation.

These methods provide reference points for evaluating batch correction:
- Best case: Perfect cell type separation
- Worst case: No correction or batch-specific processing
- Random baselines: Shuffled embeddings
"""
import numpy as np
import scanpy as sc
from typing import Optional
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.logs_ import get_logger

logger = get_logger()


def _randomize_features(X: np.ndarray, partition: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Randomize features, optionally within partitions.
    
    Args:
        X: Feature matrix (n_cells, n_features)
        partition: Optional partition labels. If provided, randomization happens within each partition.
    
    Returns:
        Randomized feature matrix
    """
    X_out = X.copy()
    if partition is None:
        partition = np.full(X.shape[0], 0)
    else:
        partition = np.asarray(partition)
    
    for partition_name in np.unique(partition):
        partition_idx = np.argwhere(partition == partition_name).flatten()
        X_out[partition_idx] = X[np.random.permutation(partition_idx)]
    
    return X_out


def _perfect_embedding(partition: np.ndarray, jitter: float = 0.01) -> np.ndarray:
    """
    Create a perfect embedding based on partition labels (one-hot encoding).
    
    Args:
        partition: Partition labels (e.g., cell types)
        jitter: Amount of random jitter to add (None for no jitter)
    
    Returns:
        Embedding matrix (n_cells, n_partitions)
    """
    le = LabelEncoder()
    encoded = le.fit_transform(partition)[:, None]
    ohe = OneHotEncoder(sparse_output=False)
    embedding = ohe.fit_transform(encoded)
    
    if jitter is not None and jitter > 0:
        embedding = embedding + np.random.uniform(-jitter, jitter, embedding.shape)
    
    return embedding.astype(np.float32)


def create_baseline_embedding(
    adata: AnnData,
    method: str,
    batch_key: str = 'batch',
    label_key: str = 'label',
    use_rep: str = 'X',
    n_comps: int = 50,
    random_seed: int = 42
) -> np.ndarray:
    """
    Create a baseline embedding for batch effects evaluation.
    
    Args:
        adata: AnnData object with data
        method: Baseline method name. Options:
            - 'no_integration': PCA on full dataset (no batch correction)
            - 'no_integration_batch': PCA computed separately per batch (worst case)
            - 'embed_cell_types': Perfect embedding based on cell types (best case for biology)
            - 'embed_cell_types_jittered': Perfect embedding with jitter
            - 'shuffle_integration': Randomize all features
            - 'shuffle_integration_by_batch': Randomize features within batches
            - 'shuffle_integration_by_cell_type': Randomize features within cell types
        batch_key: Key in adata.obs for batch labels
        label_key: Key in adata.obs for cell type/biological labels
        use_rep: Representation to use ('X', 'X_pca', or layer name)
        n_comps: Number of PCA components (for PCA-based methods)
        random_seed: Random seed for reproducibility
    
    Returns:
        Embedding matrix (n_cells, n_dims)
    """
    np.random.seed(random_seed)
    
    # Get data matrix
    if use_rep == 'X':
        X = adata.X
    elif use_rep in adata.obsm:
        X = adata.obsm[use_rep]
    elif use_rep in adata.layers:
        X = adata.layers[use_rep]
    else:
        raise ValueError(f"Representation '{use_rep}' not found in adata")
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    
    if method == 'no_integration':
        # PCA on full dataset (no batch correction)
        logger.info('Creating baseline: PCA on full dataset (no_integration)')
        adata_temp = adata.copy()
        if use_rep == 'X':
            sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        else:
            # If already using PCA, just use it
            if use_rep == 'X_pca' and use_rep in adata.obsm:
                return adata.obsm[use_rep][:, :n_comps]
            sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'no_integration_batch':
        # PCA computed separately per batch (worst case - preserves batch structure)
        logger.info('Creating baseline: PCA per batch (no_integration_batch)')
        embedding = np.zeros((adata.shape[0], n_comps), dtype=np.float32)
        batches = adata.obs[batch_key].unique()
        
        for batch in batches:
            batch_idx = adata.obs[batch_key] == batch
            n_batch_cells = np.sum(batch_idx)
            batch_adata = adata[batch_idx].copy()
            
            # Need at least 2 cells and 2 features to compute PCA
            n_features = batch_adata.n_vars
            min_dim = min(n_batch_cells, n_features)
            
            if min_dim < 2:
                # Cannot compute PCA with fewer than 2 samples or features
                # Use zero embedding for this batch (already initialized)
                logger.warning(
                    f"Batch '{batch}' has only {n_batch_cells} cells and {n_features} features. "
                    f"Skipping PCA for this batch (using zeros)."
                )
                continue
            
            # Determine number of components (must be < min(n_samples, n_features))
            n_comps_batch = min(n_comps, min_dim - 1)
            
            # Use appropriate solver based on data size
            if n_batch_cells <= n_comps or n_comps_batch < n_comps:
                solver = 'full'
            else:
                solver = 'arpack'
            
            try:
                sc.tl.pca(batch_adata, n_comps=n_comps_batch, svd_solver=solver, random_state=random_seed)
                embedding[batch_idx, :n_comps_batch] = batch_adata.obsm['X_pca']
            except Exception as e:
                logger.warning(f"PCA failed for batch '{batch}': {e}. Using zeros.")
                continue
        
        return embedding
    
    elif method == 'embed_cell_types':
        # Perfect embedding based on cell types (best case for biological signal)
        logger.info('Creating baseline: Perfect embedding based on cell types')
        return _perfect_embedding(adata.obs[label_key], jitter=None)
    
    elif method == 'embed_cell_types_jittered':
        # Perfect embedding with jitter
        logger.info('Creating baseline: Perfect embedding with jitter')
        return _perfect_embedding(adata.obs[label_key], jitter=0.01)
    
    elif method == 'shuffle_integration':
        # Randomize all features (random baseline)
        logger.info('Creating baseline: Shuffled features (random)')
        X_shuffled = _randomize_features(X)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'shuffle_integration_by_batch':
        # Randomize features within each batch (preserves batch structure)
        logger.info('Creating baseline: Shuffled features within batches')
        X_shuffled = _randomize_features(X, partition=adata.obs[batch_key].values)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'shuffle_integration_by_cell_type':
        # Randomize features within each cell type (preserves cell type structure)
        logger.info('Creating baseline: Shuffled features within cell types')
        X_shuffled = _randomize_features(X, partition=adata.obs[label_key].values)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    else:
        raise ValueError(f"Unknown baseline method: {method}. "
                         f"Options: 'no_integration', 'no_integration_batch', 'embed_cell_types', "
                         f"'embed_cell_types_jittered', 'shuffle_integration', "
                         f"'shuffle_integration_by_batch', 'shuffle_integration_by_cell_type'")


# List of available baseline methods
BASELINE_METHODS = [
    'no_integration',
    'no_integration_batch',
    'embed_cell_types',
    'embed_cell_types_jittered',
    'shuffle_integration',
    'shuffle_integration_by_batch',
    'shuffle_integration_by_cell_type'
]
