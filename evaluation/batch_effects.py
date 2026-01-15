"""
Batch effects evaluation module.

Evaluates how well embeddings remove batch effects while preserving biological signal.
"""
from pathlib import Path
from typing import Dict
from anndata import AnnData
import numpy as np
import scanpy as sc
import scib
import pandas as pd
from scipy import sparse
from utils.logs_ import get_logger
from scib.metrics import silhouette, silhouette_batch, pcr

logger = get_logger()


def compute_lisi_pure_python(
    adata: AnnData,
    obs_key: str,
    n_neighbors: int = 90,
    use_rep: str = None,
) -> np.ndarray:
    """
    Compute Local Inverse Simpson's Index (LISI) using pure Python.
    
    This implementation uses scanpy's neighbor computation to avoid 
    the scib compiled extension that requires newer GLIBC versions.
    
    LISI measures the effective number of categories (batches or cell types)
    in each cell's local neighborhood. Higher values indicate better mixing.
    
    Args:
        adata: AnnData object (must have neighbors computed in adata.obsp)
        obs_key: Key in adata.obs containing categorical labels (batch or cell type)
        n_neighbors: Number of neighbors to consider for LISI computation
        use_rep: Embedding key to use for neighbor computation (if neighbors not precomputed)
    
    Returns:
        Array of LISI scores per cell
    """
    # Get the connectivity matrix from precomputed neighbors
    if 'connectivities' not in adata.obsp:
        if use_rep is None:
            raise ValueError("Neighbors not computed. Either compute neighbors first or provide use_rep.")
        logger.info(f"Computing neighbors for LISI using {use_rep}")
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    
    # Get connectivity matrix and convert to distances (1 - connectivity for weighted graphs)
    conn = adata.obsp['connectivities']
    if sparse.issparse(conn):
        conn = conn.toarray()
    
    # Get labels
    labels = adata.obs[obs_key].values
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[l] for l in labels])
    
    n_cells = adata.n_obs
    lisi_scores = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Get neighbors for this cell (non-zero connections)
        neighbor_weights = conn[i, :]
        neighbor_mask = neighbor_weights > 0
        
        if np.sum(neighbor_mask) == 0:
            # No neighbors, LISI is 1 (only the cell itself)
            lisi_scores[i] = 1.0
            continue
        
        # Get neighbor labels and weights
        neighbor_labels = label_indices[neighbor_mask]
        weights = neighbor_weights[neighbor_mask]
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Compute proportion of each category in neighborhood
        proportions = np.zeros(n_labels)
        for j, label_idx in enumerate(neighbor_labels):
            proportions[label_idx] += weights[j]
        
        # Compute Simpson's Index: sum of squared proportions
        simpson_index = np.sum(proportions ** 2)
        
        # LISI = 1 / Simpson's Index (Inverse Simpson's Index)
        if simpson_index > 0:
            lisi_scores[i] = 1.0 / simpson_index
        else:
            lisi_scores[i] = 1.0
    
    return lisi_scores


class BatchEffectsEvaluator:
    """
    Evaluates batch effect correction in embeddings.
    
    Computes metrics that measure:
    - How well batches are mixed (iLISI, kBET)
    - How well biological signal is preserved despite batch mixing (ASW_label/batch, cLISI)
    - Overall batch effect removal (ASW_batch, PCR_batch)
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        batch_key: str = 'batch',
        label_key: str = 'label',
        save_dir: str = None,
        auto_subsample: bool = True,
        metric: str = 'euclidean',
        **kwargs
    ):
        """
        Initialize batch effects evaluator.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            batch_key: Key in adata.obs containing batch labels
            label_key: Key in adata.obs containing cell type/biological labels
            save_dir: Directory to save results
            auto_subsample: Whether to subsample large datasets (>10k cells)
            metric: Distance metric for neighbor computation ('euclidean' or 'cosine').
                   Use 'cosine' for embeddings on hypersphere (e.g., normalized embeddings).
            **kwargs: Additional parameters (e.g., n_neighbors for LISI)
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.batch_key = batch_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.auto_subsample = auto_subsample
        self.metric = metric
        self.n_neighbors = kwargs.get('n_neighbors', 90)
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate batch effects in embeddings.
        
        Returns:
            Dictionary of batch effect metrics.
        """
        logger.info(f'Evaluating batch effects for embedding: {self.embedding_key}')
        
        # Subsample if needed
        if self.auto_subsample and self.adata.shape[0] > 10000:
            logger.info('Subsampling dataset for batch effects evaluation (>10k cells)')
            from utils.sampling import sample_adata
            adata_eval = sample_adata(self.adata, sample_size=5000, stratify_by=None)
        else:
            adata_eval = self.adata.copy()
        
        # Check if multiple batches exist
        n_batches = adata_eval.obs[self.batch_key].nunique()
        if n_batches <= 1:
            logger.warning(f'Only {n_batches} batch(es) found. Batch effects evaluation requires multiple batches.')
            return {'error': 'Insufficient batches for evaluation'}
        
        results_dict = {}
        
        # Ensure neighbors are computed on the correct embedding
        if "neighbors" in adata_eval.uns:
            logger.info(f"Recomputing neighbors for embedding: {self.embedding_key}")
            adata_eval.uns.pop("neighbors", None)
        
        sc.pp.neighbors(
            adata_eval,
            use_rep=self.embedding_key,
            n_neighbors=15,
            metric=self.metric,
            random_state=0
        )
        
        # 1. ASW_batch: Average Silhouette Width for batch
        # Lower is better (less batch separation)
        logger.info('Computing ASW_batch...')
        try:
            results_dict["ASW_batch"] = silhouette(
                adata_eval,
                self.batch_key,
                self.embedding_key,
                self.metric
            )
        except Exception as e:
            logger.warning(f"ASW_batch computation failed: {e}")
            results_dict["ASW_batch"] = np.nan
        
        # 2. ASW_label/batch: Silhouette width balancing label separation and batch mixing
        # Higher is better (good label separation, good batch mixing)
        logger.info('Computing ASW_label/batch...')
        try:
            results_dict["ASW_label/batch"] = silhouette_batch(
                adata_eval,
                self.batch_key,
                self.label_key,
                embed=self.embedding_key,
                metric=self.metric,
                return_all=False,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"ASW_label/batch computation failed: {e}")
            results_dict["ASW_label/batch"] = np.nan
        
        # 3. PCR_batch: Principal Component Regression on batch
        # Lower is better (less batch variance in PCA space)
        logger.info('Computing PCR_batch...')
        try:
            results_dict["PCR_batch"] = pcr(
                adata_eval,
                covariate=self.batch_key,
                embed=self.embedding_key,
                recompute_pca=True,
                n_comps=50,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"PCR_batch computation failed: {e}")
            results_dict["PCR_batch"] = np.nan
        
        # 4. iLISI: Integration Local Inverse Simpson's Index
        # Measures batch mixing - higher is better (better mixing)
        logger.info('Computing iLISI...')
        try:
            # Use pure Python implementation to avoid GLIBC dependency
            ilisi_scores = compute_lisi_pure_python(
                adata_eval,
                obs_key=self.batch_key,
                n_neighbors=self.n_neighbors,
            )
            ilisi = np.nanmedian(ilisi_scores)
            # Normalize to [0, 1] where 1 = perfect mixing
            results_dict["iLISI"] = (ilisi - 1) / (n_batches - 1) if n_batches > 1 else np.nan
        except Exception as e:
            logger.warning(f"iLISI computation failed: {e}")
            results_dict["iLISI"] = np.nan
        
        # 5. cLISI: Cell type Local Inverse Simpson's Index
        # Measures cell type separation - lower raw LISI is better (cells cluster by type)
        # After normalization: higher is better
        logger.info('Computing cLISI...')
        try:
            # Use pure Python implementation to avoid GLIBC dependency
            clisi_scores = compute_lisi_pure_python(
                adata_eval,
                obs_key=self.label_key,
                n_neighbors=self.n_neighbors,
            )
            n_labels = adata_eval.obs[self.label_key].nunique()
            clisi = np.nanmedian(clisi_scores)
            # Normalize to [0, 1] where 1 = perfect separation
            results_dict["cLISI"] = (n_labels - clisi) / (n_labels - 1) if n_labels > 1 else np.nan
        except Exception as e:
            logger.warning(f"cLISI computation failed: {e}")
            results_dict["cLISI"] = np.nan
        
        # 6. kBET: k-nearest neighbor batch effect test
        # Measures batch mixing - higher is better (better mixing, less batch effect)
        # NOTE: kBET requires R to be installed and properly configured
        logger.info('Computing kBET...')
        try:
            # Try to import kBET - it requires R
            from scib.metrics import kBET
            results_dict["kBET"] = kBET(
                adata_eval,
                batch_key=self.batch_key,
                label_key=self.label_key,
                type_="embed",
                embed=self.embedding_key,
                scaled=True,
                verbose=False,
            )
        except ImportError:
            logger.warning(
                "kBET requires R and rpy2 to be installed. "
                "Install R and run: pip install rpy2. Skipping kBET metric."
            )
            results_dict["kBET"] = np.nan
        except Exception as e:
            error_msg = str(e)
            if "libR.so" in error_msg or "cannot open shared object" in error_msg:
                logger.warning(
                    "kBET failed: R shared library not found. "
                    "Ensure R is installed and LD_LIBRARY_PATH includes R lib directory. "
                    "You may need: export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH"
                )
            else:
                logger.warning(f"kBET computation failed: {error_msg}")
            results_dict["kBET"] = np.nan
        
        # Compute summary score (average of normalized metrics)
        # Higher is better overall
        valid_metrics = [v for v in results_dict.values() if not np.isnan(v)]
        if valid_metrics:
            # Normalize metrics to [0, 1] scale where higher is better
            # For ASW_batch and PCR_batch, lower is better, so we invert them
            normalized_scores = []
            for key, value in results_dict.items():
                if not np.isnan(value):
                    if key in ["ASW_batch", "PCR_batch"]:
                        # Invert: lower original value = higher normalized score
                        # Simple inversion (assuming values are in reasonable range)
                        normalized_scores.append(1.0 / (1.0 + value))
                    else:
                        # Already normalized or higher is better
                        normalized_scores.append(value)
            
            if normalized_scores:
                results_dict["batch_effects_score"] = np.mean(normalized_scores)
        
        # Remove NaN values
        results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
        
        # Save results to dataset's uns (unstructured metadata)
        # Use 'baseline_batch_effects' for baseline methods, 'batch_effects' for method-specific
        if self.embedding_key.startswith('X_baseline_'):
            uns_key = 'baseline_batch_effects'
        else:
            uns_key = 'batch_effects'
        
        if uns_key not in self.adata.uns:
            self.adata.uns[uns_key] = {}
        self.adata.uns[uns_key][self.embedding_key] = results_dict
        logger.info(f"Saved batch effects results to adata.uns['{uns_key}']['{self.embedding_key}']")
        
        # Also save to file if save_dir is provided
        # Append to existing file or create new one (don't overwrite previous embeddings)
        if self.save_dir:
            metrics_file = self.save_dir / 'batch_effects_metrics.csv'
            new_row = pd.DataFrame.from_dict({self.embedding_key: results_dict}, orient='index')
            
            if metrics_file.exists():
                try:
                    existing_df = pd.read_csv(metrics_file, index_col=0)
                    # Remove existing row for this embedding if present (update)
                    if self.embedding_key in existing_df.index:
                        existing_df = existing_df.drop(self.embedding_key)
                    # Concatenate with new row
                    combined_df = pd.concat([existing_df, new_row])
                    combined_df.to_csv(metrics_file)
                except Exception as e:
                    logger.warning(f"Error appending to existing file, overwriting: {e}")
                    new_row.to_csv(metrics_file)
            else:
                new_row.to_csv(metrics_file)
            logger.info(f"Saved batch effects results to '{metrics_file}'")
        
        logger.info(f"Batch effects evaluation complete. Metrics: {results_dict}")
        return results_dict
