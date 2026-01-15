"""
Script to run evaluation and classification on pre-computed embeddings.

This script loads existing embeddings from data.h5ad files in output directories
and runs evaluation and classification without re-extracting embeddings.

Usage:
    cd /lotterlab/users/riccardo/ML_BIO/SCFM_meta
    uv run python /lotterlab/users/riccardo/ML_BIO/scFM_repos/scFM_eval/run/run_eval_classify_from_cache.py <config_path>
"""
import sys
import os
from pathlib import Path
from os.path import dirname, abspath, join, exists
import anndata as ad
import yaml

# Add scFM_eval to path
dir_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, dir_path)

from run.run_exp import Experiment, set_random_seed
from setup_path import OUTPUT_PATH, PARAMS_PATH
from utils.logs_ import get_logger

logger = get_logger()


def load_embeddings_from_cache(experiment: Experiment) -> bool:
    """
    Load embeddings from cached data.h5ad file.
    
    Args:
        experiment: Experiment instance with save_dir and embedding_key set
        
    Returns:
        True if embeddings were loaded successfully, False otherwise
    """
    cached_file = join(experiment.save_dir, 'data.h5ad')
    
    if not exists(cached_file):
        logger.warning(f"Cached embeddings not found at {cached_file}")
        return False
    
    logger.info(f"Loading cached embeddings from {cached_file}")
    cached_adata = ad.read_h5ad(cached_file)
    
    # Check if embedding key exists in cached data
    if experiment.embedding_key not in cached_adata.obsm:
        logger.warning(f"Embedding key '{experiment.embedding_key}' not found in cached data")
        logger.info(f"Available obsm keys: {list(cached_adata.obsm.keys())}")
        return False
    
    # Check if cell counts match (basic sanity check)
    if cached_adata.n_obs != experiment.loader.adata.n_obs:
        logger.warning(
            f"Cell count mismatch: cached data has {cached_adata.n_obs} cells, "
            f"but loaded data has {experiment.loader.adata.n_obs} cells. "
            f"This may indicate data filtering differences."
        )
        # Try to match by index if possible
        common_idx = experiment.loader.adata.obs.index.intersection(cached_adata.obs.index)
        if len(common_idx) == experiment.loader.adata.n_obs:
            logger.info(f"Found matching indices, using subset of cached embeddings")
            experiment.loader.adata.obsm[experiment.embedding_key] = cached_adata[common_idx].obsm[experiment.embedding_key].copy()
        else:
            logger.error("Cannot match cells between cached and loaded data")
            return False
    else:
        # Transfer embeddings to the loaded data
        logger.info(f"Transferring embeddings '{experiment.embedding_key}' to loaded data")
        experiment.loader.adata.obsm[experiment.embedding_key] = cached_adata.obsm[experiment.embedding_key].copy()
    
    # Also transfer other obsm keys if they exist (like UMAP)
    for key in cached_adata.obsm.keys():
        if key not in experiment.loader.adata.obsm:
            experiment.loader.adata.obsm[key] = cached_adata.obsm[key].copy()
            logger.info(f"Also transferred '{key}' from cached data")
    
    logger.info(f"Successfully loaded embeddings with shape {experiment.loader.adata.obsm[experiment.embedding_key].shape}")
    return True


def run_eval_and_classify(config_path: str):
    """
    Run evaluation and classification on pre-computed embeddings.
    
    Args:
        config_path: Path to YAML config file (relative to PARAMS_PATH)
    """
    set_random_seed(42)
    
    # Create experiment instance
    experiment = Experiment(config_path)
    
    logger.info("=" * 70)
    logger.info(f"Running evaluation and classification for: {config_path}")
    logger.info(f"Output directory: {experiment.save_dir}")
    logger.info("=" * 70)
    
    # Load data (needed for splits and labels)
    logger.info("Step 1: Loading data...")
    experiment.load_data()
    
    # Skip QC and preprocessing (already done)
    logger.info("Step 2: Skipping QC (data already preprocessed)")
    experiment.qc_data()  # Will skip due to config
    
    logger.info("Step 3: Skipping preprocessing (data already preprocessed)")
    experiment.preprocess_data()  # Will skip due to config
    
    # Set embedding key from config (must be done before loading cache)
    from run.run_exp import embedding_method_map
    feat_config = experiment.feat_config
    experiment.embedding_key = embedding_method_map[feat_config['method']]
    logger.info(f"Embedding method: {feat_config['method']} -> key: {experiment.embedding_key}")
    
    # Load embeddings from cache
    logger.info("Step 4: Loading embeddings from cache...")
    success = load_embeddings_from_cache(experiment)
    if not success:
        logger.error(f"Failed to load cached embeddings. Please check that embeddings exist at {join(experiment.save_dir, 'data.h5ad')}")
        return False
    
    # Run visualization (optional, but embeddings are already visualized)
    if experiment.vis_embedding:
        logger.info("Step 5: Skipping visualization (already done)")
        # experiment.visualize_embedding()  # Skip since already done
    
    # Run evaluation
    if experiment.eval_embedding:
        logger.info("Step 6: Running embedding evaluation...")
        try:
            experiment.evaluate_embedding()
            logger.info("✓ Embedding evaluation completed")
        except Exception as e:
            logger.error(f"✗ Embedding evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run classification
    logger.info("Step 7: Running classification...")
    try:
        experiment.train_classifier()
        logger.info("✓ Classification completed")
    except Exception as e:
        logger.error(f"✗ Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("=" * 70)
    logger.info("✓ All steps completed successfully!")
    logger.info("=" * 70)
    return True


def main():
    """
    Main entry point.
    """
    if len(sys.argv) < 2:
        print("Usage: python run_eval_classify_from_cache.py <config_path>")
        print("Example: python run_eval_classify_from_cache.py brca_full/cell_type/scimilarity.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = run_eval_and_classify(config_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

