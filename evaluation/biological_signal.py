"""
Biological signal evaluation module.

Evaluates how well embeddings preserve biological structure (cell types, clusters).
"""
from pathlib import Path
from typing import Dict
from anndata import AnnData
import numpy as np
import scanpy as sc
import scib
import pandas as pd
from utils.logs_ import get_logger

logger = get_logger()


class BiologicalSignalEvaluator:
    """
    Evaluates biological signal preservation in embeddings.
    
    Computes metrics that measure:
    - Clustering quality (NMI, ARI)
    - Cell type separation (ASW_label)
    - Graph connectivity (preservation of cell type neighborhoods)
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        label_key: str = 'label',
        save_dir: str = None,
        auto_subsample: bool = True,
        metric: str = 'euclidean',
        leiden_resolution: float = 0.3,
        n_neighbors: int = 15,
        **kwargs
    ):
        """
        Initialize biological signal evaluator.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            label_key: Key in adata.obs containing cell type/biological labels
            save_dir: Directory to save results
            auto_subsample: Whether to subsample large datasets (>10k cells)
            metric: Distance metric for neighbor computation ('euclidean' or 'cosine')
            leiden_resolution: Resolution parameter for Leiden clustering
            n_neighbors: Number of neighbors for graph construction
            **kwargs: Additional parameters
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.auto_subsample = auto_subsample
        self.metric = metric
        self.leiden_resolution = leiden_resolution
        self.n_neighbors = n_neighbors
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate biological signal preservation in embeddings.
        
        Returns:
            Dictionary of biological signal metrics.
        """
        logger.info(f'Evaluating biological signal for embedding: {self.embedding_key}')
        
        # Subsample if needed
        if self.auto_subsample and self.adata.shape[0] > 10000:
            logger.info('Subsampling dataset for biological signal evaluation (>10k cells)')
            from utils.sampling import sample_adata
            adata_eval = sample_adata(self.adata, sample_size=5000, stratify_by=None)
        else:
            adata_eval = self.adata.copy()
        
        results_dict = {}
        
        # Ensure neighbors are computed on the correct embedding
        if "neighbors" in adata_eval.uns:
            logger.info(f"Recomputing neighbors for embedding: {self.embedding_key}")
            adata_eval.uns.pop("neighbors", None)
        
        # Build k-nearest neighbor graph
        sc.pp.neighbors(
            adata_eval,
            use_rep=self.embedding_key,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=0
        )
        
        # Perform Leiden clustering
        logger.info(f'Computing Leiden clustering (resolution={self.leiden_resolution})...')
        sc.tl.leiden(
            adata_eval,
            key_added="cluster",
            resolution=self.leiden_resolution,
            random_state=0
        )
        
        # 1. NMI_cluster/label: Normalized Mutual Information
        # Measures agreement between clusters and true labels
        logger.info('Computing NMI_cluster/label...')
        try:
            results_dict["NMI_cluster/label"] = scib.metrics.nmi(
                adata_eval,
                "cluster",
                self.label_key,
                "arithmetic",
                nmi_dir=None
            )
        except Exception as e:
            logger.warning(f"NMI computation failed: {e}")
            results_dict["NMI_cluster/label"] = np.nan
        
        # 2. ARI_cluster/label: Adjusted Rand Index
        # Measures agreement between clusters and true labels
        logger.info('Computing ARI_cluster/label...')
        try:
            results_dict["ARI_cluster/label"] = scib.metrics.ari(
                adata_eval,
                "cluster",
                self.label_key
            )
        except Exception as e:
            logger.warning(f"ARI computation failed: {e}")
            results_dict["ARI_cluster/label"] = np.nan
        
        # 3. ASW_label: Average Silhouette Width for labels
        # Measures cell type separation in embedding space
        logger.info('Computing ASW_label...')
        try:
            results_dict["ASW_label"] = scib.metrics.silhouette(
                adata_eval,
                self.label_key,
                self.embedding_key,
                self.metric
            )
        except Exception as e:
            logger.warning(f"ASW_label computation failed: {e}")
            results_dict["ASW_label"] = np.nan
        
        # 4. graph_conn: Graph connectivity
        # Measures fraction of cells that remain connected to their cell type
        logger.info('Computing graph_conn...')
        try:
            results_dict["graph_conn"] = scib.metrics.graph_connectivity(
                adata_eval,
                label_key=self.label_key
            )
        except Exception as e:
            logger.warning(f"graph_conn computation failed: {e}")
            results_dict["graph_conn"] = np.nan
        
        # Compute summary score (average of biological metrics)
        valid_metrics = [
            results_dict.get("NMI_cluster/label"),
            results_dict.get("ARI_cluster/label"),
            results_dict.get("ASW_label")
        ]
        valid_metrics = [v for v in valid_metrics if not np.isnan(v)]
        if valid_metrics:
            results_dict["avg_bio"] = np.mean(valid_metrics)
        
        # Remove NaN values
        results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
        
        # Save results to dataset's uns (unstructured metadata)
        if 'biological_signal' not in self.adata.uns:
            self.adata.uns['biological_signal'] = {}
        self.adata.uns['biological_signal'][self.embedding_key] = results_dict
        logger.info(f"Saved biological signal results to adata.uns['biological_signal']['{self.embedding_key}']")
        
        # Also save to file if save_dir is provided
        # Append to existing file or create new one (don't overwrite previous embeddings)
        if self.save_dir:
            metrics_file = self.save_dir / 'biological_signal_metrics.csv'
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
            logger.info(f"Saved biological signal results to '{metrics_file}'")
        
        logger.info(f"Biological signal evaluation complete. Metrics: {results_dict}")
        return results_dict
