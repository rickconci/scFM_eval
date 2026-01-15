"""
scConcept embedding extractor for scFM_eval framework.
Extracts embeddings using pre-trained scConcept models.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor

logger = get_logger()

# Try to import scConcept
try:
    from concept import scConcept
    SCCONCEPT_AVAILABLE = True
except ImportError:
    logger.warning(
        "scConcept not available. Install with: pip install git+https://github.com/theislab/scConcept.git@main"
    )
    SCCONCEPT_AVAILABLE = False


class scConceptExtractor(EmbeddingExtractor):
    """
    Extractor for scConcept embeddings.
    
    Extracts cell embeddings using pre-trained scConcept models.
    Stores embeddings in adata.obsm['X_scConcept'].
    """
    
    def __init__(self, params):
        """
        Initialize scConcept extractor.
        
        Args:
            params: Dictionary containing:
                - method: 'scConcept'
                - params: Dictionary with:
                    - model_name: Model name (e.g., 'Corpus-30M')
                    - cache_dir: Directory for cached models (default: './cache/')
                    - batch_size: Batch size for embedding extraction (default: 32)
                    - gene_id_column: Column name in adata.var for gene IDs (default: None, uses index)
                    - repo_id: HuggingFace repository ID (default: 'theislab/scConcept')
        """
        super().__init__(params)
        logger.info(f'scConceptExtractor ({self.params})')
        
        if not SCCONCEPT_AVAILABLE:
            raise ImportError(
                "scConcept is not available. Please install it with: "
                "pip install git+https://github.com/theislab/scConcept.git@main"
            )
        
        # Extract parameters with defaults
        self.model_name = self.params.get('model_name', 'Corpus-30M')
        self.cache_dir = Path(self.params.get('cache_dir', './cache/'))
        self.batch_size = self.params.get('batch_size', 32)
        self.gene_id_column = self.params.get('gene_id_column', None)
        self.repo_id = self.params.get('repo_id', 'theislab/scConcept')
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scConcept instance (model will be loaded in fit_transform)
        self.concept = None
        self._model_loaded = False
        
        logger.info(
            f"scConcept extractor initialized: "
            f"model_name={self.model_name}, cache_dir={self.cache_dir}, "
            f"batch_size={self.batch_size}, gene_id_column={self.gene_id_column}"
        )
    
    @staticmethod
    def validate_config(params):
        """
        Validate configuration parameters.
        
        Args:
            params: Configuration dictionary
        """
        assert 'params' in params and params['params'] is not None, "Missing 'params' in parameters"
        assert 'model_name' in params['params'], "Missing 'model_name' in parameters"
    
    def _load_model(self):
        """
        Load scConcept model if not already loaded.
        """
        if self._model_loaded and self.concept is not None:
            return
        
        logger.info(f"Loading scConcept model: {self.model_name}")
        
        # Initialize scConcept
        self.concept = scConcept(
            repo_id=self.repo_id,
            cache_dir=str(self.cache_dir)
        )
        
        # Load model
        self.concept.load_config_and_model(model_name=self.model_name)
        
        self._model_loaded = True
        logger.info(f"scConcept model {self.model_name} loaded successfully")
    
    def fit_transform(self, data_loader):
        """
        Extract embeddings from AnnData using scConcept.
        
        Note: scConcept's model applies log1p internally during batch processing.
        If data is already normalized+log1p (preprocessed=True), this will result
        in double log1p transformation. However, scConcept's API doesn't provide
        a way to skip this, so we log a warning.
        
        Args:
            data_loader: Data loader object with adata attribute
            
        Returns:
            numpy.ndarray: Cell embeddings of shape (n_cells, embedding_dim)
        """
        if not SCCONCEPT_AVAILABLE:
            raise ImportError("scConcept is not available")
        
        data = data_loader.adata
        logger.info(f"Extracting scConcept embeddings from AnnData with shape {data.shape}")
        
        # Check if data is already preprocessed
        # scConcept's model applies log1p internally, so it expects normalized (TP10K) but NOT log1p
        if hasattr(data, 'uns') and data.uns.get('preprocessed', False):
            preprocessing_method = data.uns.get('preprocessing_method', 'unknown')
            if 'log1p' in preprocessing_method.lower():
                logger.warning(
                    "Data is already preprocessed with log1p transformation. "
                    "scConcept's model applies log1p internally during batch processing, "
                    "which will result in double log1p transformation. "
                    "The preprocessing step should have skipped log1p for scConcept - please check configuration."
                )
            else:
                logger.info(
                    f"Data is preprocessed ({preprocessing_method}). "
                    "scConcept will apply log1p internally during batch processing."
                )
        
        # Load model if not already loaded
        self._load_model()
        
        # Use Ensembl IDs from saved column (SCConcept requires Ensembl IDs)
        if 'gene_id' in data.var.columns:
            data.var.index = data.var['gene_id'].values
            gene_id_column_to_use = None  # Use var.index
        else:
            gene_id_column_to_use = self.gene_id_column
        
        # Extract embeddings
        logger.info(
            f"Extracting embeddings with parameters: "
            f"batch_size={self.batch_size}, gene_id_column={gene_id_column_to_use}"
        )
        
        result = self.concept.extract_embeddings(
            adata=data,
            batch_size=self.batch_size,
            gene_id_column=gene_id_column_to_use
        )
        
        # Store CLS embeddings in adata.obsm
        embeddings = result['cls_cell_emb']
        data.obsm['X_scConcept'] = embeddings
        
        logger.info(
            f"Extracted embeddings with shape {embeddings.shape} "
            f"and stored in adata.obsm['X_scConcept']"
        )
        
        return embeddings.copy()

