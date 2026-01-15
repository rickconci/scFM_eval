from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import anndata as ad
import scanpy as sc
import logging
logger = logging.getLogger('ml_logger')
from utils.logs_ import get_logger

from utils.sampling import sample_adata
logger = get_logger()

# Keys that are cached for PCA/UMAP
PCA_UMAP_KEYS = ['X_pca', 'X_umap', 'X_pca_umap']


class EmbeddingVisualizer:
    """Class for visualizing embeddings with optional batch and label coloring."""
    def __init__(self, embedding, obs, save_dir=".", auto_subsample=True):
        
        self.auto_subsample=auto_subsample
        self.embedding = embedding
            
        self.obs = obs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_pca_umap_from_cache(self, adata_original, cache_path):
        """
        Load cached PCA/UMAP and align with current adata by cell index.
        
        Args:
            adata_original: AnnData object to load PCA/UMAP into
            cache_path: Path to the cached PCA/UMAP h5ad file
            
        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False
        
        try:
            logger.info(f'Loading cached PCA/UMAP from: {cache_path}')
            cached_adata = ad.read_h5ad(cache_path)
            
            # Check if cell indices match (allowing for QC filtering)
            cached_cells = set(cached_adata.obs.index)
            current_cells = set(adata_original.obs.index)
            
            # Check if current cells are a subset of cached cells
            if not current_cells.issubset(cached_cells):
                logger.warning('Current cells are not a subset of cached cells. Recomputing PCA/UMAP.')
                return False
            
            # Align by cell index and load into adata
            for key in PCA_UMAP_KEYS:
                if key in cached_adata.obsm:
                    # Align by cell index
                    adata_original.obsm[key] = cached_adata[adata_original.obs.index].obsm[key]
                    logger.info(f'Loaded {key} from cache (shape: {adata_original.obsm[key].shape})')
            
            return True
            
        except Exception as e:
            logger.warning(f'Failed to load PCA/UMAP cache: {e}. Recomputing.')
            return False

    def save_pca_umap_to_cache(self, adata_original, cache_path):
        """
        Save PCA/UMAP to a lightweight cache file.
        
        Saves only cell indices and obsm entries (PCA, UMAP) for efficient reuse.
        
        Args:
            adata_original: AnnData object with computed PCA/UMAP
            cache_path: Path to save the cache file
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create lightweight adata with just cell indices and PCA/UMAP obsm entries
        cache_adata = ad.AnnData(obs=adata_original.obs[[]])  # Empty obs, keeps index
        
        for key in PCA_UMAP_KEYS:
            if key in adata_original.obsm:
                cache_adata.obsm[key] = adata_original.obsm[key]
        
        # Add metadata
        cache_adata.uns['cache_info'] = {
            'n_cells': adata_original.n_obs,
            'keys': list(cache_adata.obsm.keys()),
        }
        
        cache_adata.write_h5ad(cache_path)
        logger.info(f'Saved PCA/UMAP cache to: {cache_path}')
        logger.info(f'  Cached keys: {list(cache_adata.obsm.keys())}')

    def compute_pca_umap(self, adata_original):
        '''
        Compute PCA and UMAP on original adata.
        This is dataset-dependent, not method-dependent.
        
        Computes:
        1. X_pca: PCA on original expression data
        2. X_umap: UMAP on original expression data (neighbors on X)
        3. X_pca_umap: UMAP on PCA of original expression data (neighbors on X_pca)
        '''
        # 1. Compute PCA on original expression data (if not already computed)
        if 'X_pca' not in adata_original.obsm:
            logger.info('Computing PCA on original expression data (dataset-dependent)')
            sc.tl.pca(adata_original, n_comps=50)
            logger.info('Saved PCA to X_pca in original adata')
        else:
            logger.info('PCA on original data already exists in dataset, skipping computation')
        
        # 2. Compute UMAP on original expression data X (if not already computed)
        if 'X_umap' not in adata_original.obsm:
            logger.info('Computing UMAP on original expression data X (dataset-dependent)')
            sc.pp.neighbors(adata_original, use_rep='X')
            sc.tl.umap(adata_original)
            logger.info('Saved UMAP to X_umap in original adata (neighbors on X)')
        else:
            logger.info('UMAP on original data (X_umap) already exists in dataset, skipping computation')
        
        # 3. Compute UMAP on PCA of original expression data (if not already computed)
        if 'X_pca_umap' not in adata_original.obsm:
            if 'X_pca' in adata_original.obsm:
                logger.info('Computing UMAP on PCA of original expression data (dataset-dependent)')
                existing_X_umap = adata_original.obsm.get('X_umap', None)
                sc.pp.neighbors(adata_original, use_rep='X_pca')
                sc.tl.umap(adata_original)
                adata_original.obsm['X_pca_umap'] = adata_original.obsm['X_umap'].copy()
                if existing_X_umap is not None:
                    adata_original.obsm['X_umap'] = existing_X_umap
                else:
                    adata_original.obsm.pop('X_umap', None)
                logger.info('Saved UMAP to X_pca_umap in original adata (neighbors on X_pca)')
            else:
                logger.warning('Cannot compute X_pca_umap: X_pca not found')
        else:
            logger.info('UMAP on PCA data (X_pca_umap) already exists in dataset, skipping computation')

    def save_pca_umap_to_adata(self, adata_original, cache_path=None):
        '''
        Load or compute PCA and UMAP on original adata, with optional caching.
        
        If cache_path is provided:
        1. Try to load from cache
        2. If cache doesn't exist or is incompatible, compute and save to cache
        
        Args:
            adata_original: AnnData object to compute/load PCA/UMAP for
            cache_path: Optional path to cache file for PCA/UMAP
        '''
        # Try to load from cache first
        if cache_path is not None:
            if self.load_pca_umap_from_cache(adata_original, cache_path):
                logger.info('Successfully loaded PCA/UMAP from cache')
                return
        
        # Compute PCA/UMAP
        self.compute_pca_umap(adata_original)
        
        # Save to cache if path provided
        if cache_path is not None:
            self.save_pca_umap_to_cache(adata_original, cache_path)

    def plot(self, adata_original=None, embedding_key=None, save_to_adata=True, pca_umap_cache_path=None):
        """
        Generate and save embedding visualizations.
        
        Computes (or loads from cache):
        1. PCA on original expression data (X) -> saved as 'X_pca'
        2. UMAP on original expression data (neighbors on X) -> saved as 'X_umap'
        3. UMAP on PCA of original expression data (neighbors on X_pca) -> saved as 'X_pca_umap'
        4. UMAP on method embeddings -> saved as 'X_umap_{embedding_key}'
        
        Args:
            adata_original: Original AnnData object to save PCA/UMAP to (optional)
            embedding_key: Key of the embedding in adata_original.obsm (optional)
            save_to_adata: If True and adata_original provided, save PCA/UMAP to original adata
            pca_umap_cache_path: Path to cache PCA/UMAP (dataset-specific, reused across methods)
        """
        if save_to_adata and adata_original is not None:
            # Compute or load PCA/UMAP (dataset-dependent, not method-dependent)
            self.save_pca_umap_to_adata(adata_original, cache_path=pca_umap_cache_path)
            
        # 3. Compute UMAP on method embeddings for visualization
        adata_embedding = ad.AnnData(X=self.embedding)
        adata_embedding.obs = self.obs.copy()
        logger.info(f'Visualizing embeddings {adata_embedding.X.shape}')

        if self.auto_subsample:
            if adata_embedding.shape[0] > 10000:
                adata_embedding = sample_adata(adata_embedding, sample_size=5000, stratify_by=None)
        
        # Compute neighbors and UMAP on embeddings
        logger.info('Computing neighbors and UMAP on method embeddings')
        sc.pp.neighbors(adata_embedding, use_rep='X')
        sc.tl.umap(adata_embedding)
        
        # Save UMAP on embeddings to original adata if requested
        if save_to_adata and adata_original is not None and embedding_key is not None:
            umap_embedding_key = f'X_umap_{embedding_key}'
            # Compute on full dataset (not subsampled)
            if adata_original.shape[0] == self.embedding.shape[0]:
                adata_embedding_full = ad.AnnData(X=self.embedding)
                adata_embedding_full.obs = adata_original.obs.copy()
                sc.pp.neighbors(adata_embedding_full, use_rep='X')
                sc.tl.umap(adata_embedding_full)
                adata_original.obsm[umap_embedding_key] = adata_embedding_full.obsm['X_umap']
                logger.info(f'Saved UMAP on embeddings to {umap_embedding_key} in original adata')
            else:
                logger.warning(f'Cannot save full UMAP: embedding shape {self.embedding.shape[0]} != adata shape {adata_original.shape[0]}')

        # Generate plots using subsampled data for visualization
        if 'batch' in adata_embedding.obs.columns:
            embeddings_fig = sc.pl.umap(adata_embedding, color='batch', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_batch.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by batch to 'embedding_batch.png'")

        if 'label' in adata_embedding.obs.columns:
            embeddings_fig = sc.pl.umap(adata_embedding, color='label', show=False, wspace=0.4, frameon=False, return_fig=True)
            embeddings_fig.savefig(self.save_dir / 'embedding_label.png', dpi=200, bbox_inches='tight')
            logger.info("Saved UMAP plot colored by label to 'embedding_label.png'")

