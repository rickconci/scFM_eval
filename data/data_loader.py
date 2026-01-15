"""
data_loader.py

Data loading utilities for single-cell data, supporting CSV and H5AD formats with preprocessing and filtering.
"""

import os
import json
import logging

import pandas as pd
import anndata as ad
import scanpy as sc

from utils.logs_ import get_logger
from setup_path import BASE_PATH, DATA_PATH
from os.path import join, isabs
import numpy as np
import h5py
from typing import Dict, Any, Tuple
from scipy.sparse import issparse
import scipy.stats


class DataLoader:
    """Base class for loading and preprocessing single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for data loading.
        """
        self.params = params
        # Resolve path: if relative, join with DATA_PATH; if absolute, use as-is
        path = params['path']
        if isabs(path):
            self.path = path
        else:
            self.path = join(DATA_PATH, path)
        self.dataset_name = os.path.basename(self.path).split(".")[0]
        self.layer = params['layer_name']
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.train_test_split = params['train_test_split']
        self.cv_splits = params['cv_splits']
        self.log = get_logger()
        self.adata = None

    @staticmethod
    def validate_config(params):
        """Validate that required parameters are present.

        Args:
            params (dict): Configuration parameters.
        Raises:
            AssertionError: If required parameters are missing.
        """
        assert 'path' in params, "Missing required parameter: 'path'"

    def prepare_data(self, process_dir):
        """Prepare data for downstream analysis. To be implemented in subclasses.

        Args:
            process_dir (str): Directory for processing outputs.
        """
        pass

    def _extract_h5ad_metadata(self) -> Dict[str, Any]:
        """Extract metadata from H5AD file using h5py for efficient partial reading.
        
        Extracts metadata for obs, var, uns, obsm, varm, layers, obsp, and varp
        without loading the full data matrices.
        
        Returns:
            Dict[str, Any]: Metadata dictionary containing:
                - obs: DataFrame with cell metadata
                - var: DataFrame with gene metadata
                - uns: Unstructured metadata dictionary
                - obsm: Dictionary with keys and shapes of observation multidimensional arrays
                - varm: Dictionary with keys and shapes of variable multidimensional arrays
                - layers: Dictionary with keys and shapes of layer data
                - obsp: Dictionary with keys and shapes of observation pairwise arrays
                - varp: Dictionary with keys and shapes of variable pairwise arrays
                - shape: Tuple of (n_obs, n_vars)
        """
        self.log.info(f"Extracting metadata from H5AD file: {self.path}")
        
        metadata: Dict[str, Any] = {
            'file_path': self.path,
            'obs': None,
            'var': None,
            'uns': None,
            'obsm': {},
            'varm': {},
            'layers': {},
            'obsp': {},
            'varp': {},
            'shape': None,
            'obs_columns': [],
            'var_columns': []
        }
        
        try:
            with h5py.File(self.path, 'r') as f:
                from anndata.io import read_elem
                
                # Read obs (cell metadata) - typically much smaller than X
                if 'obs' in f:
                    obs_data = read_elem(f['obs'])
                    if isinstance(obs_data, pd.DataFrame):
                        metadata['obs'] = obs_data
                        metadata['obs_columns'] = list(obs_data.columns)
                    else:
                        self.log.warning("obs is not a DataFrame, skipping")
                
                # Read var (gene metadata)
                if 'var' in f:
                    var_data = read_elem(f['var'])
                    if isinstance(var_data, pd.DataFrame):
                        metadata['var'] = var_data
                        metadata['var_columns'] = list(var_data.columns)
                    else:
                        self.log.warning("var is not a DataFrame, skipping")
                
                # Read uns (unstructured metadata)
                if 'uns' in f:
                    metadata['uns'] = read_elem(f['uns'])
                
                # Read obsm (observations multidimensional arrays)
                if 'obsm' in f:
                    for key in f['obsm'].keys():
                        try:
                            arr = f['obsm'][key]
                            if hasattr(arr, 'shape'):
                                metadata['obsm'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading obsm['{key}']: {e}")
                
                # Read varm (variables multidimensional arrays)
                if 'varm' in f:
                    for key in f['varm'].keys():
                        try:
                            arr = f['varm'][key]
                            if hasattr(arr, 'shape'):
                                metadata['varm'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading varm['{key}']: {e}")
                
                # Read layers (layered data)
                if 'layers' in f:
                    for key in f['layers'].keys():
                        try:
                            arr = f['layers'][key]
                            if hasattr(arr, 'shape'):
                                metadata['layers'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading layers['{key}']: {e}")
                
                # Read obsp (observations pairwise arrays)
                if 'obsp' in f:
                    for key in f['obsp'].keys():
                        try:
                            arr = f['obsp'][key]
                            if hasattr(arr, 'shape'):
                                metadata['obsp'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading obsp['{key}']: {e}")
                
                # Read varp (variables pairwise arrays)
                if 'varp' in f:
                    for key in f['varp'].keys():
                        try:
                            arr = f['varp'][key]
                            if hasattr(arr, 'shape'):
                                metadata['varp'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading varp['{key}']: {e}")
                
                # Get shape information without loading X
                if 'obs' in f and 'var' in f:
                    n_obs = f['obs'].attrs.get('_index', None)
                    if n_obs is None:
                        # Fallback: try to get length from obs DataFrame
                        if metadata['obs'] is not None:
                            n_obs = len(metadata['obs'])
                        else:
                            # Last resort: check shape attribute if available
                            n_obs = f.attrs.get('n_obs', None)
                    
                    n_vars = f['var'].attrs.get('_index', None)
                    if n_vars is None:
                        if metadata['var'] is not None:
                            n_vars = len(metadata['var'])
                        else:
                            n_vars = f.attrs.get('n_vars', None)
                    
                    if n_obs is not None and n_vars is not None:
                        metadata['shape'] = (n_obs, n_vars)
                    elif 'shape' in f.attrs:
                        metadata['shape'] = tuple(f.attrs['shape'])
                
                # Check preprocessing status
                is_preprocessed = False
                preprocessing_method = None
                if metadata['uns'] and isinstance(metadata['uns'], dict):
                    is_preprocessed = metadata['uns'].get('preprocessed', False)
                    preprocessing_method = metadata['uns'].get('preprocessing_method', None)
                
                # Log summary of extracted metadata
                log_parts = [
                    f"Shape: {metadata['shape']}",
                    f"Obs columns: {len(metadata['obs_columns'])}",
                    f"Var columns: {len(metadata['var_columns'])}"
                ]
                
                # Add preprocessing status
                if is_preprocessed:
                    method_str = f" ({preprocessing_method})" if preprocessing_method else ""
                    log_parts.append(f"Preprocessed: True{method_str}")
                else:
                    log_parts.append("Preprocessed: False")
                
                if metadata['obsm']:
                    log_parts.append(f"obsm keys: {list(metadata['obsm'].keys())}")
                if metadata['varm']:
                    log_parts.append(f"varm keys: {list(metadata['varm'].keys())}")
                if metadata['layers']:
                    log_parts.append(f"layers keys: {list(metadata['layers'].keys())}")
                if metadata['obsp']:
                    log_parts.append(f"obsp keys: {list(metadata['obsp'].keys())}")
                if metadata['varp']:
                    log_parts.append(f"varp keys: {list(metadata['varp'].keys())}")
                
                self.log.info(f"Metadata extracted. {', '.join(log_parts)}")
                
        except Exception as e:
            self.log.error(f"Error extracting metadata from H5AD file: {e}")
            raise
        
        return metadata

    def load(self):
        """Load data and return AnnData object. To be implemented in subclasses.

        Returns:
            ad.AnnData: Loaded data object.
        """
        print(f"Loading data from {self.path}")
        adata = ad.AnnData()
        self.adata = adata
        self.log.info(f'Data Loaded, {self.adata.X.shape}')
        self.log.info(
            f'min {np.min(self.adata.X)}, max {np.max(self.adata.X)}')
        return adata

    def extract_dataset_metadata(self) -> Dict[str, Any]:
        """Extract metadata from dataset without loading full data.
        
        Returns:
            Dict[str, Any]: Metadata dictionary with obsm keys, preprocessing status, and other info.
        """
        if hasattr(self, 'path') and self.path.endswith('.h5ad'):
            return self._extract_h5ad_metadata()
        else:
            # For non-H5AD files, return minimal metadata
            return {'obsm': {}, 'obs_columns': [], 'var_columns': [], 'uns': {}}
    
    def is_preprocessed(self) -> bool:
        """Check if dataset is already preprocessed (normalized + log1p).
        
        Returns:
            bool: True if dataset is marked as preprocessed, False otherwise.
        """
        metadata = self.extract_dataset_metadata()
        if metadata.get('uns') and isinstance(metadata['uns'], dict):
            return metadata['uns'].get('preprocessed', False)
        return False

    def _filter(self, adata, filter_dict):
        """Filter AnnData object based on filter_dict.

        Args:
            adata (ad.AnnData): AnnData object to filter.
            filter_dict (dict): Dictionary of {column: [values]} to filter by.
        Returns:
            ad.AnnData: Filtered AnnData object.
        """
        # Build combined mask to filter only once (avoids multiple slow view creations)
        mask = np.ones(adata.n_obs, dtype=bool)
        for col, values in filter_dict.items():
            mask &= adata.obs[col].isin(values).values
        
        # Return view to avoid expensive copy of large X matrix
        return adata[mask]

    def _translate_split_ids(self):
        """Translate split IDs to match data format.

        Handles mismatch where CV splits use 'Post' but data has 'On' in donor_id_pre_post.
        Translates 'BIOKEY_X_Post' -> 'BIOKEY_X_On' in split IDs to match actual data.
        """
        if not hasattr(self, 'cv_split_dict') or not hasattr(self, 'train_test_split_dict'):
            return

        # Quick check: only translate if id_column is donor_id_pre_post and splits contain '_Post'
        needs_translation = False
        if self.cv_split_dict:
            id_column = self.cv_split_dict.get('id_column', '')
            if id_column == 'donor_id_pre_post':
                # Quick check: sample first few split IDs to see if they have '_Post'
                fold1 = self.cv_split_dict.get('fold_1', {})
                sample_ids = fold1.get('train_ids', [])[
                    :3] if fold1.get('train_ids') else []
                if sample_ids and any('_Post' in str(sid) for sid in sample_ids):
                    # Check data format quickly (sample only, don't load all unique values)
                    if 'donor_id_pre_post' in self.adata.obs.columns:
                        # Sample a small subset to check format (much faster than .unique())
                        sample_data = self.adata.obs['donor_id_pre_post'].head(
                            1000).astype(str)
                        has_on_in_data = any('_On' in str(dv)
                                             for dv in sample_data.values)
                        if has_on_in_data:
                            needs_translation = True
                            self.log.info(
                                "Translating split IDs: '_Post' -> '_On' to match data format"
                            )

        if needs_translation:
            # Translate train_test_split
            if self.train_test_split_dict:
                split_dict = self.train_test_split_dict.get(
                    'train_test_split', {})
                if 'train_ids' in split_dict:
                    # Optimized: only translate IDs that contain '_Post' (in-place)
                    train_ids = split_dict['train_ids']
                    for idx, sid in enumerate(train_ids):
                        if '_Post' in str(sid):
                            train_ids[idx] = str(sid).replace('_Post', '_On')
                if 'test_ids' in split_dict:
                    # Optimized: only translate IDs that contain '_Post' (in-place)
                    test_ids = split_dict['test_ids']
                    for idx, sid in enumerate(test_ids):
                        if '_Post' in str(sid):
                            test_ids[idx] = str(sid).replace('_Post', '_On')

            # Translate CV splits
            if self.cv_split_dict:
                n_splits = self.cv_split_dict.get('n_splits', 0)
                for i in range(1, n_splits + 1):
                    fold_key = f'fold_{i}'
                    if fold_key in self.cv_split_dict:
                        fold = self.cv_split_dict[fold_key]
                        if 'train_ids' in fold:
                            # Optimized: only translate IDs that contain '_Post' (in-place)
                            train_ids = fold['train_ids']
                            for idx, sid in enumerate(train_ids):
                                if '_Post' in str(sid):
                                    train_ids[idx] = str(
                                        sid).replace('_Post', '_On')
                        if 'test_ids' in fold:
                            # Optimized: only translate IDs that contain '_Post' (in-place)
                            test_ids = fold['test_ids']
                            for idx, sid in enumerate(test_ids):
                                if '_Post' in str(sid):
                                    test_ids[idx] = str(
                                        sid).replace('_Post', '_On')

    def qc(self, min_genes, min_cells):
        """Apply quality control filters to the data.

        Args:
            min_genes (int): Minimum number of genes per cell.
            min_cells (int): Minimum number of cells per gene.
        """
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.log.info(
            f"Applying QC with min_genes={self.min_genes}, min_cells={self.min_cells}")
        self.log.info(f'obs shape before filtering, {self.adata.X.shape}')

        sc.pp.calculate_qc_metrics(
            self.adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(self.adata, min_genes=int(self.min_genes))
        sc.pp.filter_genes(self.adata, min_cells=int(self.min_cells))
        self.log.info(f'obs shape after filtering, {self.adata.X.shape}')

    def _check_if_normalized(self, sample_size: int = 10000) -> Tuple[bool, bool]:
        """
        Check if data is already normalized/log1p transformed based on distribution.
        
        Args:
            sample_size: Number of cells to sample for distribution check.
            
        Returns:
            tuple: (is_normalized, is_log1p) - whether data appears normalized and/or log1p transformed.
        """
        # Sample data for analysis
        n_cells = min(sample_size, self.adata.n_obs)
        if issparse(self.adata.X):
            sample = self.adata.X[:n_cells].toarray()
        else:
            sample = self.adata.X[:n_cells]
        
        # Flatten for statistics
        sample_flat = sample.flatten()
        
        # Calculate statistics
        mean_val = np.mean(sample_flat)
        median_val = np.median(sample_flat)
        max_val = np.max(sample_flat)
        zero_pct = np.sum(sample_flat == 0) / sample_flat.size * 100
        skew_val = scipy.stats.skew(sample_flat)
        
        self.log.info(f"Distribution check - Mean: {mean_val:.3f}, Median: {median_val:.3f}, "
                     f"Max: {max_val:.3f}, Zeros: {zero_pct:.1f}%, Skew: {skew_val:.3f}")
        
        # Heuristics for normalized data:
        # 1. Max value < 15-20 (log1p of normalized counts typically < 10)
        # 2. Values are floats (not integers)
        # 3. Distribution is more compressed (though zeros can still cause high skew)
        
        is_log1p = max_val < 15  # Log1p normalized data typically has max < 10-15
        is_normalized = max_val < 50  # Normalized (but not log) typically < 50
        
        # Additional check: if values are integers and large, definitely raw counts
        if not issparse(self.adata.X):
            sample_int = sample[:1000]  # Check smaller sample for integer check
            is_integer = np.allclose(sample_int, np.round(sample_int))
            if is_integer and max_val > 100:
                is_normalized = False
                is_log1p = False
                self.log.info("Data appears to be raw counts (integers, large values)")
        
        return is_normalized, is_log1p

    def plot_expression_distribution(self, save_dir, title_prefix="", sample_size=10000):
        """
        Plot distribution of expression values (.X) before and/or after normalization.
        
        Args:
            save_dir: Directory to save the plot
            title_prefix: Prefix for plot title
            sample_size: Number of cells to sample for plotting
        """
        from pathlib import Path
        from matplotlib import pyplot as plt
        from scipy.sparse import issparse
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample data
        n_cells = min(sample_size, self.adata.n_obs)
        if issparse(self.adata.X):
            sample = self.adata.X[:n_cells].toarray()
        else:
            sample = self.adata.X[:n_cells]
        
        # Flatten for histogram
        sample_flat = sample.flatten()
        
        # Remove zeros for better visualization (log scale)
        sample_nonzero = sample_flat[sample_flat > 0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Histogram (linear scale, including zeros)
        axes[0].hist(sample_flat, bins=100, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Expression Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{title_prefix}Distribution (all values)')
        axes[0].axvline(np.mean(sample_flat), color='r', linestyle='--', label=f'Mean: {np.mean(sample_flat):.3f}')
        axes[0].axvline(np.median(sample_flat), color='g', linestyle='--', label=f'Median: {np.median(sample_flat):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram (log scale, non-zero values only)
        if len(sample_nonzero) > 0:
            axes[1].hist(sample_nonzero, bins=100, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Expression Value (log scale)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'{title_prefix}Distribution (non-zero values, log scale)')
            axes[1].set_xscale('log')
            axes[1].axvline(np.mean(sample_nonzero), color='r', linestyle='--', label=f'Mean: {np.mean(sample_nonzero):.3f}')
            axes[1].axvline(np.median(sample_nonzero), color='g', linestyle='--', label=f'Median: {np.median(sample_nonzero):.3f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No non-zero values', ha='center', va='center')
            axes[1].set_title(f'{title_prefix}Distribution (non-zero values)')
        
        # Add statistics text
        stats_text = (
            f'Shape: {self.adata.shape}\n'
            f'Max: {np.max(sample_flat):.3f}\n'
            f'Mean: {np.mean(sample_flat):.3f}\n'
            f'Median: {np.median(sample_flat):.3f}\n'
            f'Zeros: {np.sum(sample_flat == 0) / len(sample_flat) * 100:.1f}%'
        )
        fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # Clean filename: remove spaces, convert to lowercase, replace special chars
        filename_prefix = title_prefix.lower().replace(" ", "_").replace("-", "_").strip("_")
        plot_path = save_dir / f'{filename_prefix}expression_distribution.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log.info(f'Saved expression distribution plot to {plot_path}')

    def scale(self, normalize, target_sum, apply_log1p, method=None, plots_dir=None, skip_distribution_check=False):
        """Normalize and/or log-transform the data.
        
        Args:
            normalize (bool): Whether to normalize total counts per cell.
            target_sum (float): Target sum for normalization.
            apply_log1p (bool): Whether to apply log1p transformation.
            method (str, optional): Embedding method name (e.g., 'scimilarity', 'scconcept').
                                   Used to apply method-specific preprocessing.
            plots_dir (str, optional): Directory to save distribution plots.
            skip_distribution_check (bool): If True, skip the distribution-based normalization check.
                                           Use when caller has already performed the check.
        """
        self.normalize = normalize
        self.target_sum = target_sum
        self.apply_log1p = apply_log1p
        self.preprocessing_method = method

        # Check if preprocessing is already done (metadata flag)
        if hasattr(self.adata, 'uns') and self.adata.uns.get('preprocessed', False):
            self.log.info('Data already preprocessed (metadata flag), skipping...')
            # Still plot distribution if plots_dir provided
            if plots_dir:
                self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")
            return
        
        # Plot pre-normalization distribution if plots_dir provided
        if plots_dir:
            self.plot_expression_distribution(plots_dir, title_prefix="Pre-normalization ")
        
        # Check if data is already normalized based on distribution (unless caller already checked)
        if not skip_distribution_check:
            is_normalized, is_log1p_detected = self._check_if_normalized()
            
            # Skip normalization if data appears already normalized
            if self.normalize and is_normalized:
                self.log.warning(f'Data appears already normalized (max value < 50). Skipping normalization step.')
                self.normalize = False  # Don't normalize again
            
            # Skip log1p if data appears already log1p transformed
            if self.apply_log1p and is_log1p_detected:
                self.log.warning(f'Data appears already log1p transformed (max value < 15). Skipping log1p step.')
                self.apply_log1p = False  # Don't log1p again
            
            # If both steps are skipped, mark as preprocessed and return
            if not self.normalize and not self.apply_log1p:
                self.log.info('Both normalization and log1p appear already done. Marking as preprocessed.')
                if not hasattr(self.adata, 'uns'):
                    self.adata.uns = {}
                self.adata.uns['preprocessed'] = True
                self.adata.uns['preprocessing_method'] = 'detected_normalized'
                # Plot distribution
                if plots_dir:
                    self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")
                return

        if self.normalize:
            self.log.info(f'Normalizing data (target_sum={self.target_sum})... This may take a while for large datasets.')
            sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
            self.log.info('Normalization complete.')
        if self.apply_log1p:
            self.log.info('Applying log1p transformation... This may take a while for large datasets.')
            sc.pp.log1p(self.adata)
            self.log.info('Log1p transformation complete.')
        
        # Always check and log distribution after processing
        self.log.info('Checking distribution after preprocessing...')
        is_normalized_post, is_log1p_post = self._check_if_normalized()
        
        # Mark as preprocessed
        if not hasattr(self.adata, 'uns'):
            self.adata.uns = {}
        self.adata.uns['preprocessed'] = True
        
        # Set descriptive preprocessing method name
        if method:
            self.adata.uns['preprocessing_method'] = method
        else:
            # Create descriptive name based on what was actually done
            method_parts = []
            if self.normalize:
                method_parts.append(f'tp{int(self.target_sum)}k')
            if self.apply_log1p:
                method_parts.append('log1p')
            self.adata.uns['preprocessing_method'] = '_'.join(method_parts) if method_parts else 'none'
        
        self.log.info(
            f"Applied preprocessing to data.X, apply_log1p = {self.apply_log1p}, target_sum = {self.target_sum}")
        self.log.info(f'Data shape after preprocessing: {self.adata.X.shape}')
        
        # Plot post-normalization distribution if plots_dir provided
        if plots_dir:
            self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")

    def hvg(self, n_top_genes, flavor, batch_key=None):
        """Select highly variable genes (HVGs).

        Args:
            n_top_genes (int): Number of top HVGs to select.
            flavor (str): Method for HVG selection (e.g., 'seurat').
            batch_key (str, optional): Batch key for batch-aware HVG selection.
        """
        # Ensure batch column is categorical if batch_key is provided
        if batch_key is not None and batch_key in self.adata.obs.columns:
            if not pd.api.types.is_categorical_dtype(self.adata.obs[batch_key]):
                self.log.info(f"Converting batch column '{batch_key}' to categorical dtype")
                self.adata.obs[batch_key] = self.adata.obs[batch_key].astype('category')
        
        sc.pp.highly_variable_genes(
            self.adata, batch_key=batch_key, flavor=flavor, subset=True, n_top_genes=n_top_genes)
        self.log.info(
            f"Applied HVG to data.X, n_top_genes = {n_top_genes}, flavor = {flavor}, batch_key = {batch_key}")
        self.log.info(f'Data shape after HVG, {self.adata.X.shape}')


    def ensure_both_gene_identifiers(self, gene_symbol_column: str = 'feature_name', normalize: bool = True):
        """Ensure both Ensembl IDs and gene symbols are available in var columns.
        
        This method ensures that var has both:
        - 'gene_id' column: Contains Ensembl IDs (for SCConcept)
        - 'gene_symbol' column: Contains gene symbols (for SCimilarity)
        
        The original var.index is preserved (typically Ensembl IDs).
        
        Args:
            gene_symbol_column (str): Column name in var that contains gene symbols.
                                     Default: 'feature_name'
            normalize (bool): If True, normalize gene symbols (uppercase, strip whitespace).
                             Default: True
        """
        if self.adata is None:
            raise ValueError("Data must be loaded before ensuring gene identifiers")
        
        def normalize_symbol(symbol: str) -> str:
            """Normalize gene symbol: uppercase and strip whitespace."""
            if normalize:
                return str(symbol).strip().upper()
            return str(symbol).strip()
        
        # Store original index (likely Ensembl IDs)
        original_index = self.adata.var.index.copy()
        
        # Determine what we have
        has_gene_symbols = gene_symbol_column in self.adata.var.columns
        sample_indices = original_index[:min(100, len(original_index))].astype(str)
        index_is_ensembl = sum(1 for idx in sample_indices if idx.startswith('ENSG') or idx.startswith('ENSMUSG')) > len(sample_indices) * 0.8
        
        # Ensure 'gene_id' column (Ensembl IDs)
        if 'gene_id' not in self.adata.var.columns:
            if index_is_ensembl:
                # var.index is Ensembl IDs, use it
                self.adata.var['gene_id'] = original_index.astype(str)
                self.log.info("Added 'gene_id' column from var.index (Ensembl IDs)")
            elif has_gene_symbols:
                # var.index might be gene symbols, need to find Ensembl IDs
                # Check if there's an Ensembl ID column
                possible_ensembl_cols = [col for col in self.adata.var.columns 
                                        if any(x in str(col).lower() for x in ['ensembl', 'ensg', 'gene_id'])]
                if possible_ensembl_cols:
                    self.adata.var['gene_id'] = self.adata.var[possible_ensembl_cols[0]].astype(str)
                    self.log.info(f"Added 'gene_id' column from '{possible_ensembl_cols[0]}'")
                else:
                    # Try to convert gene symbols back to Ensembl IDs (reverse lookup)
                    self.log.warning("Could not find Ensembl IDs. 'gene_id' column not created.")
            else:
                self.log.warning("Could not determine Ensembl IDs. 'gene_id' column not created.")
        else:
            self.log.info("'gene_id' column already exists")
        
        # Ensure 'gene_symbol' column (gene symbols)
        if 'gene_symbol' not in self.adata.var.columns:
            if has_gene_symbols:
                # Use the gene_symbol_column
                symbols = self.adata.var[gene_symbol_column].astype(str)
                # Normalize and handle duplicates
                seen = {}
                normalized_symbols = []
                for symbol in symbols:
                    normalized = normalize_symbol(symbol)
                    if normalized in seen:
                        seen[normalized] += 1
                        normalized_symbols.append(f"{normalized}__dup{seen[normalized]}")
                    else:
                        seen[normalized] = 0
                        normalized_symbols.append(normalized)
                self.adata.var['gene_symbol'] = normalized_symbols
                self.log.info(f"Added 'gene_symbol' column from '{gene_symbol_column}' (normalized: {normalize})")
            elif index_is_ensembl:
                # var.index is Ensembl IDs, convert to gene symbols
                self.log.info("Converting Ensembl IDs to gene symbols using mygene...")
                try:
                    import mygene
                except ImportError:
                    raise ImportError(
                        "mygene is required for Ensembl ID conversion. "
                        "Install it with: pip install mygene"
                    )
                
                mg = mygene.MyGeneInfo()
                ensembl_ids = original_index.astype(str).tolist()
                batch_size = 1000
                id_to_symbol = {}
                
                for i in range(0, len(ensembl_ids), batch_size):
                    batch = ensembl_ids[i:i + batch_size]
                    try:
                        results = mg.querymany(
                            batch,
                            scopes='ensembl.gene',
                            fields='symbol',
                            species='human',
                            returnall=False,
                            as_dataframe=False
                        )
                        for result in results:
                            ensembl_id = result.get('query', '')
                            symbol = result.get('symbol', '')
                            if symbol:
                                id_to_symbol[ensembl_id] = symbol
                            else:
                                id_to_symbol[ensembl_id] = ensembl_id
                    except Exception as e:
                        self.log.warning(f"Error converting batch {i//batch_size + 1}: {e}")
                        for ensembl_id in batch:
                            if ensembl_id not in id_to_symbol:
                                id_to_symbol[ensembl_id] = ensembl_id
                
                # Normalize and handle duplicates
                symbols = [id_to_symbol.get(ensembl_id, ensembl_id) for ensembl_id in ensembl_ids]
                seen = {}
                normalized_symbols = []
                for symbol in symbols:
                    normalized = normalize_symbol(symbol)
                    if normalized in seen:
                        seen[normalized] += 1
                        normalized_symbols.append(f"{normalized}__dup{seen[normalized]}")
                    else:
                        seen[normalized] = 0
                        normalized_symbols.append(normalized)
                
                self.adata.var['gene_symbol'] = normalized_symbols
                self.log.info(f"Added 'gene_symbol' column from Ensembl ID conversion (normalized: {normalize})")
            else:
                # Assume var.index is already gene symbols
                normalized_symbols = [normalize_symbol(symbol) for symbol in original_index.astype(str)]
                self.adata.var['gene_symbol'] = normalized_symbols
                self.log.info("Added 'gene_symbol' column from var.index (normalized: {normalize})")
        else:
            self.log.info("'gene_symbol' column already exists")
        
        # Summary
        has_gene_id = 'gene_id' in self.adata.var.columns
        has_gene_symbol = 'gene_symbol' in self.adata.var.columns
        self.log.info(f"Gene identifiers available: gene_id={has_gene_id}, gene_symbol={has_gene_symbol}")
    
    def ensure_gene_symbols_in_var_index(self, gene_column: str = 'feature_name', normalize: bool = True):
        """Ensure var.index contains gene symbols (for SCimilarity compatibility).
        
        This is a convenience method that:
        1. Calls ensure_both_gene_identifiers() to ensure both are available
        2. Sets var.index to gene symbols from 'gene_symbol' column
        
        Args:
            gene_column (str): Column name in var that contains gene symbols (used if gene_symbol not available).
                              Default: 'feature_name'
            normalize (bool): If True, normalize gene symbols (uppercase, strip whitespace).
                             Default: True
        """
        # First ensure both identifiers are available
        self.ensure_both_gene_identifiers(gene_symbol_column=gene_column, normalize=normalize)
        
        # Then set var.index to gene symbols
        if 'gene_symbol' in self.adata.var.columns:
            self.adata.var.index = self.adata.var['gene_symbol'].values
            self.log.info("Set var.index to gene symbols from 'gene_symbol' column")
        else:
            self.log.warning("'gene_symbol' column not available, var.index unchanged")


class CSVDataLoader(DataLoader):
    """Loader for CSV-formatted single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for CSV loading.
        """
        super().__init__(params)
        self.path = params['path']

    def load(self):
        """Load data from a CSV file.

        Returns:
            ad.AnnData: Loaded data object.
        """
        self.log.info(f"Loading CSV data from {self.path}")
        df = pd.read_csv(self.path)
        labels = df['label'].values
        features = df.drop(columns=['label']).values
        adata = ad.AnnData(X=features, obs=pd.DataFrame({'label': labels}))
        self.adata = adata
        return adata


class H5ADLoader(DataLoader):
    """Loader for H5AD-formatted single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for H5AD loading.
        """
        super().__init__(params)
        # Path is already resolved in base class __init__
        self.load_raw = bool(params['load_raw'])
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.filter = params.get('filter', None)

    def load(self):
        """Load data from an H5AD file, with optional filtering and splitting.

        Returns:
            ad.AnnData: Loaded data object.
        """
        logging.info(f"Loading H5AD data from {self.path}")
        self.log.info("Reading H5AD file (this may take a while for large files)...")
        
        adata = ad.read_h5ad(self.path)
        self.log.info(f"H5AD file loaded. Initial shape: {adata.shape}")
        
        adata.obs['label'] = adata.obs[self.label_key]
        adata.obs['batch'] = adata.obs[self.batch_key]
        
        if self.load_raw:
            self.log.info("Converting raw to adata...")
            adata = adata.raw.to_adata()

        # Preserve counts layer for SCimilarity (requires layers["counts"] for lognorm_counts)
        if 'counts' not in adata.layers:
            if self.layer == 'X':
                adata.layers['counts'] = adata.X
                self.log.info('Stored reference to X as counts layer')
            elif self.layer in adata.layers:
                self.log.info(f"Creating counts layer from {self.layer}...")
                adata.layers['counts'] = adata.layers[self.layer].copy()
                self.log.info(f'Created counts layer from {self.layer}')

        if self.layer != 'X':
            self.log.info(f"Switching to layer '{self.layer}'...")
            if 'original_X' not in adata.layers:
                adata.layers['original_X'] = adata.X
            adata.X = adata.layers[self.layer]

        self.log.info(f'Data Loaded, {adata.X.shape}')
        
        # Compute min/max efficiently
        try:
            if hasattr(adata.X, 'data') and len(adata.X.data) > 0:
                x_min = float(adata.X.data.min()) if adata.X.nnz > 0 else 0.0
                x_max = float(adata.X.data.max()) if adata.X.nnz > 0 else 0.0
            else:
                x_min = float(np.min(adata.X))
                x_max = float(np.max(adata.X))
            self.log.info(f'X min {x_min}, X max {x_max}')
        except Exception as e:
            self.log.info(f'X min/max computation skipped: {e}')

        if self.filter is not None:
            self.log.info(f"Applying filters: {self.filter}")
            filter_dict = {entry["column"]: entry["values"]
                           for entry in self.filter}
            adata = self._filter(adata, filter_dict)
            self.log.info(f"After filtering, shape: {adata.shape}")

        self.adata = adata

        if self.train_test_split:
            fname = os.path.join(BASE_PATH, self.train_test_split)
            with open(fname, 'r') as f:
                self.train_test_split_dict = json.load(f)
            fname = os.path.join(BASE_PATH, self.cv_splits)
            with open(fname, 'r') as f:
                self.cv_split_dict = json.load(f)

            # Fix mismatch: CV splits use 'Post' but data may have 'On' in donor_id_pre_post
            self._translate_split_ids()

        return adata
