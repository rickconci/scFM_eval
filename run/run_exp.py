"""
Main experiment runner for scFM_eval.
Handles configuration, data loading, preprocessing, feature extraction, and evaluation.
"""
import importlib
import sys
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import anndata as ad

dir_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, dir_path)

from viz.visualization import EmbeddingVisualizer
from utils.logs_ import set_logging
from setup_path import OUTPUT_PATH, PARAMS_PATH, EMBEDDINGS_PATH
from run.utils import timing, get_configs, get_embedding_key
from run.evaluation_runner import EvaluationRunner


class Experiment:
    """
    Handles the execution of a machine learning experiment defined by a YAML config.
    Handles loading data, preprocessing, feature extraction, and evaluation.
    """
    def __init__(self, config_path):
        """
        Initialize the Experiment with a given config path.
        Sets up directories, logging, and loads configuration.
        """
        # Handle both relative and absolute config paths
        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = join(PARAMS_PATH, config_path)
        
        self.run_id, self.data_config, self.qc_config, self.preproc_config, self.hvg, self.feat_config, self.evaluations_config = get_configs(self.config_path)

        self.vis_embedding = bool(self.feat_config.get('viz', False))
        
        # Extract dataset_name, task_name, method_name from config
        # These must be explicitly set in the YAML config
        dataset_name = self.data_config.get('dataset_name')
        task_name = self.data_config.get('task_name')
        method_name = self.feat_config.get('method', 'unknown')
        
        if not dataset_name:
            raise ValueError(
                f"Missing 'dataset_name' in YAML config: {self.config_path}. "
                f"Please add 'dataset_name: <name>' to the 'dataset' section."
            )
        if not task_name:
            raise ValueError(
                f"Missing 'task_name' in YAML config: {self.config_path}. "
                f"Please add 'task_name: <name>' to the 'dataset' section."
            )
        
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.method_name = method_name

        # Build output path: OUTPUT_PATH / task / method / dataset /
        config_filename = os.path.basename(self.config_path)
        save_dir = str(OUTPUT_PATH / task_name / method_name / dataset_name)
        self.save_dir = save_dir + f'_{self.run_id}' if self.run_id else save_dir

        # Create organized subdirectories
        self.config_dir = join(self.save_dir, 'config')
        self.logs_dir = join(self.save_dir, 'logs')
        self.plots_dir = join(self.save_dir, 'plots')
        self.metrics_dir = join(self.save_dir, 'metrics')
        
        # Create subdirectories for plots
        self.plots_embeddings_dir = join(self.plots_dir, 'embeddings')
        self.plots_distributions_dir = join(self.plots_dir, 'distributions')
        self.plots_evaluations_dir = join(self.plots_dir, 'evaluations')
        
        # Create subdirectories for metrics
        self.metrics_batch_effects_dir = join(self.metrics_dir, 'batch_effects')
        self.metrics_biological_signal_dir = join(self.metrics_dir, 'biological_signal')
        self.metrics_classification_dir = join(self.metrics_dir, 'classification')
        self.metrics_annotation_dir = join(self.metrics_dir, 'annotation')
        
        # Create all directories
        for dir_path in [self.save_dir, self.config_dir, self.logs_dir, self.plots_dir, 
                        self.metrics_dir, self.plots_embeddings_dir, 
                        self.plots_distributions_dir, self.plots_evaluations_dir, 
                        self.metrics_batch_effects_dir, self.metrics_biological_signal_dir, 
                        self.metrics_classification_dir, self.metrics_annotation_dir]:
            if not exists(dir_path):
                os.makedirs(dir_path)

        # Copy config_file.yaml to the config subdirectory
        shutil.copyfile(self.config_path, join(self.config_dir, config_filename))

        # Set logging format (save logs to logs subdirectory)
        self.log = set_logging(self.logs_dir)

        # Placeholders for data and embeddings
        self.data = None
        self.embedding = None
        self.embedding_key = None
        
        # Set up embedding cache path: EMBEDDINGS_PATH / dataset_name / method.h5ad
        # Use dataset_name from config (already set above)
        self.embeddings_dir = EMBEDDINGS_PATH / self.dataset_name
        self.embedding_cache_path = self.embeddings_dir / f'{self.method_name}.h5ad'
        
        # Set up PCA/UMAP cache path: EMBEDDINGS_PATH / dataset_name / pca_umap.h5ad
        # This is dataset-specific (not method-specific) so can be reused across methods
        self.pca_umap_cache_path = self.embeddings_dir / 'pca_umap.h5ad'

    @timing
    def run(self):
        """
        Run the complete experiment workflow.
        
        Workflow:
        1. Load data and prepare gene identifiers
        2. Check if data is normalized; normalize if needed (BEFORE embeddings)
        3. Generate embeddings for ALL cells (no QC) OR load cached embeddings
        4. Apply QC/preprocessing (for evaluations only)
        5. Run evaluations
        """
        # Step 1: Load data
        self.load_data()
        self.loader.ensure_both_gene_identifiers(normalize=True)
        self.data = self.loader.adata
        
        # Step 2: Check and normalize data BEFORE embedding generation
        # This ensures embeddings are generated from normalized data, and extractors
        # won't try to normalize again (they check adata.uns['preprocessed'] flag)
        self.log.info('Checking if data is already normalized...')
        self.check_and_normalize_for_embeddings()
        
        # Step 3: Get embeddings (generate or load from cache)
        embeddings_exist = self.check_model_embeddings_in_dataset()
        if embeddings_exist:
            self.log.info('Loading cached embeddings...')
            self.load_cached_embeddings()
        else:
            self.log.info('Generating embeddings for all cells...')
            self.generate_embeddings()
            self.save_embeddings()
        
        # Step 4: Apply QC/preprocessing (for evaluations)
        self.log.info('Applying QC and preprocessing for evaluations...')
        self.qc_data()
        self.preprocess_data()
        if self.hvg:
            self.filter_hvg()
        
        # Update embedding reference after filtering (obsm was auto-filtered with adata)
        self.embedding = self.loader.adata.obsm[self.embedding_key]
        self.log.info(f'After QC: {self.embedding.shape[0]} cells with embeddings')
        
        # Step 5: Run evaluations
        if self.vis_embedding:
            self.visualize_embedding()
        self.run_evaluations()
        self.generate_summaries()

    def load_class(self, module_path, class_name):
        """
        Dynamically import and return a class by module and name.

        Args:
            module_path (str): Python module path.
            class_name (str): Name of the class to import.

        Returns:
            type: The class object.
        """
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


    @timing
    def load_data(self):
        """Load the dataset."""
        loader_config = self.data_config.copy()
        LoaderClass = self.load_class(loader_config['module'], loader_config['class'])
        self.log.info(f'Data Loader config: {loader_config}')
        self.log.info('Loading dataset...')
        self.loader = LoaderClass(loader_config)
        self.data = self.loader.load()
        self.log.info(f'Dataset loaded successfully. Shape: {self.data.shape}')
    
    @timing
    def check_and_normalize_for_embeddings(self):
        """
        Check if data is already normalized, and normalize if needed.
        
        This happens BEFORE embedding generation to ensure:
        1. Embeddings are generated from normalized data
        2. Extractors can skip their own normalization (they check adata.uns['preprocessed'])
        
        Uses distribution-based heuristics to detect if data is already normalized/log1p.
        
        Method-specific handling:
        - scConcept: Normalize (TP10K) but skip log1p (model applies log1p internally)
        - Other methods: Normalize (TP10K) + log1p
        """
        preproc_config = self.preproc_config
        
        # Check if method requires special handling (e.g., scConcept doesn't want log1p)
        # scConcept's model applies log1p internally, so we should skip log1p here
        method_name = self.method_name.lower()
        skip_log1p_for_method = 'scconcept' in method_name or 'sc concept' in method_name
        
        # Allow YAML override: if preprocessing.apply_log1p_for_embeddings is False, skip log1p
        apply_log1p_for_embeddings = preproc_config.get('apply_log1p_for_embeddings', None)
        if apply_log1p_for_embeddings is False:
            skip_log1p_for_method = True
            self.log.info('YAML config: apply_log1p_for_embeddings=False, skipping log1p for embeddings')
        elif apply_log1p_for_embeddings is True:
            skip_log1p_for_method = False
            self.log.info('YAML config: apply_log1p_for_embeddings=True, will apply log1p for embeddings')
        
        # Check if data is already normalized using distribution heuristics
        is_normalized, is_log1p_detected = self.loader._check_if_normalized()
        
        if is_log1p_detected:
            if skip_log1p_for_method:
                self.log.info('Data appears already normalized and log1p transformed. '
                            f'Method {method_name} expects non-log data - this may cause issues.')
            else:
                self.log.info('Data appears already normalized and log1p transformed. Skipping normalization.')
            # Mark as preprocessed so extractors skip normalization
            if not hasattr(self.loader.adata, 'uns'):
                self.loader.adata.uns = {}
            self.loader.adata.uns['preprocessed'] = True
            self.loader.adata.uns['preprocessing_method'] = 'detected_normalized_log1p'
        elif is_normalized:
            # Data is normalized but not log1p
            if skip_log1p_for_method:
                self.log.info(f'Data appears already normalized (but not log1p). '
                            f'Method {method_name} expects non-log data - keeping as is.')
                # Mark as preprocessed without applying log1p
                if not hasattr(self.loader.adata, 'uns'):
                    self.loader.adata.uns = {}
                self.loader.adata.uns['preprocessed'] = True
                self.loader.adata.uns['preprocessing_method'] = 'detected_normalized_no_log1p'
            else:
                self.log.info('Data appears already normalized (but not log1p). Applying log1p only.')
                # Apply log1p only
                scale_params = {
                    'normalize': False,  # Skip normalization
                    'target_sum': preproc_config.get('target_sum', 10000),
                    'apply_log1p': True,  # Apply log1p
                    'method': None,
                    'plots_dir': None,
                    'skip_distribution_check': True  # Already checked above
                }
                self.loader.scale(**scale_params)
                # Mark as preprocessed
                if not hasattr(self.loader.adata, 'uns'):
                    self.loader.adata.uns = {}
                self.loader.adata.uns['preprocessed'] = True
                self.loader.adata.uns['preprocessing_method'] = 'normalized_centrally_log1p_added'
        else:
            # Data is not normalized - normalize and conditionally apply log1p
            if skip_log1p_for_method:
                self.log.info(f'Data appears to be raw counts. Normalizing (TP10K) but skipping log1p '
                            f'for method {method_name} (model applies log1p internally).')
                apply_log1p = False
            else:
                self.log.info('Data appears to be raw counts. Normalizing and applying log1p...')
                apply_log1p = preproc_config.get('apply_log1p', True)
            
            scale_params = {
                'normalize': preproc_config.get('normalize', True),
                'target_sum': preproc_config.get('target_sum', 10000),
                'apply_log1p': apply_log1p,
                'method': None,
                'plots_dir': None,
                'skip_distribution_check': True  # Already checked above
            }
            self.loader.scale(**scale_params)
            # Mark as preprocessed
            if not hasattr(self.loader.adata, 'uns'):
                self.loader.adata.uns = {}
            self.loader.adata.uns['preprocessed'] = True
            if apply_log1p:
                self.loader.adata.uns['preprocessing_method'] = 'normalized_centrally'
            else:
                self.loader.adata.uns['preprocessing_method'] = 'normalized_centrally_no_log1p'
        
        if skip_log1p_for_method:
            self.log.info(f'Data is ready for embedding generation (normalized, no log1p for {method_name}).')
        else:
            self.log.info('Data is ready for embedding generation (normalized + log1p).')
    
    def check_model_embeddings_in_dataset(self) -> bool:
        """
        Check if cached embeddings exist for this dataset/method combination.
        
        Returns:
            bool: True if embeddings cache file exists, False otherwise.
        """
        return self.embedding_cache_path.exists()
    
    @timing
    def load_cached_embeddings(self):
        """Load cached embeddings and align with current adata by cell index."""
        self.embedding_key = get_embedding_key(self.feat_config)
        
        self.log.info(f'Loading cached embeddings from: {self.embedding_cache_path}')
        emb_adata = ad.read_h5ad(self.embedding_cache_path)
        
        # Align by cell index (embeddings were generated for all cells)
        self.embedding = emb_adata[self.loader.adata.obs.index].obsm[self.embedding_key]
        self.loader.adata.obsm[self.embedding_key] = self.embedding
        
        self.log.info(f'Loaded embeddings. Key: {self.embedding_key}, Shape: {self.embedding.shape}')
    
    @timing
    def qc_data(self):
        """Apply quality control filters to the dataset."""
        qc_config = self.qc_config
        if ('skip' in qc_config) and (qc_config['skip'] is True):
            return
        # Filter config to only include parameters accepted by qc()
        qc_params = {
            'min_genes': qc_config.get('min_genes'),
            'min_cells': qc_config.get('min_cells')
        }
        self.loader.qc(**qc_params)

    @timing
    def preprocess_data(self):
        """
        Apply preprocessing steps for evaluations.
        
        Note: Normalization is already done before embedding generation.
        This method now only handles plotting distributions for evaluations.
        If data was already normalized, this will detect and skip re-normalization.
        """
        preproc_config = self.preproc_config
        
        self.log.info('=' * 60)
        self.log.info('Checking preprocessing status for evaluations')
        self.log.info('=' * 60)
        
        # Check if data is already preprocessed (from embedding generation step)
        if hasattr(self.loader.adata, 'uns') and self.loader.adata.uns.get('preprocessed', False):
            self.log.info('Data already preprocessed before embeddings. Skipping normalization.')
            # Still plot distribution for evaluations
            if self.plots_distributions_dir:
                self.loader.plot_expression_distribution(
                    self.plots_distributions_dir, 
                    title_prefix="Post-normalization "
                )
            return
        
        # If somehow not preprocessed, apply preprocessing now
        self.log.info('Applying preprocessing (normalization + log1p) for evaluations...')
        scale_params = {
            'normalize': preproc_config.get('normalize'),
            'target_sum': preproc_config.get('target_sum'),
            'apply_log1p': preproc_config.get('apply_log1p'),
            'method': preproc_config.get('method'),
            'plots_dir': self.plots_distributions_dir
        }
        self.loader.scale(**scale_params)
        self.log.info('Preprocessing complete.')

    @timing
    def filter_hvg(self):
        """Select High Variant Genes only, if specified in config."""
        hvg_config = self.hvg
        if ('skip' in hvg_config) and (hvg_config['skip'] is True):
            self.log.info('Skipping HVG. Analyzing all genes')
            return
        # Filter config to only include parameters accepted by hvg()
        hvg_params = {
            'n_top_genes': hvg_config.get('n_top_genes'),
            'flavor': hvg_config.get('flavor'),
            'batch_key': hvg_config.get('batch_key')
        }
        self.loader.hvg(**hvg_params)

    @timing
    def generate_embeddings(self):
        """Extract features or embeddings from the dataset."""
        self.log.info('=' * 60)
        self.log.info('Extract features or embeddings from the dataset')
        self.log.info('=' * 60)
        feat_config = self.feat_config
        
        # Extract embeddings
        if 'methods' in feat_config and isinstance(feat_config['methods'], list):
            # Multiple methods: extract each and concatenate
            self.log.info(f'Extracting embeddings from {len(feat_config["methods"])} methods')
            embeddings_list = []
            
            for method_config in feat_config['methods']:
                method_name = method_config['name']
                self.log.info(f'Extracting embeddings using {method_name}')
                method_key = f'X_{method_name.lower()}'
                
                single_feat_config = {
                    'method': method_name,
                    'module': method_config['module'],
                    'class': method_config['class'],
                    'params': method_config.get('params', {}),
                    'viz': False,
                    'eval': False,
                }
                single_feat_config['params']['save_dir'] = self.save_dir
                
                # Load and instantiate extractor
                ExtractorClass = self.load_class(single_feat_config['module'], single_feat_config['class'])
                extractor = ExtractorClass(single_feat_config)
                
                # Extract embeddings
                method_embedding = extractor.fit_transform(self.loader)
                embeddings_list.append(method_embedding)
                
                # Store individual embeddings in adata.obsm
                if method_key not in self.loader.adata.obsm:
                    self.loader.adata.obsm[method_key] = method_embedding
            
            # Join embeddings based on method
            joining_method = feat_config.get('embedding_joining_method', 'concatenate')
            if joining_method == 'concatenate':
                self.embedding = np.concatenate(embeddings_list, axis=1)
                self.log.info(f'Concatenated embeddings: {[e.shape[1] for e in embeddings_list]} -> {self.embedding.shape[1]}')
            else:
                raise ValueError(f"Unsupported embedding joining method: {joining_method}")
            
            # Set embedding key (depends on joining method)
            self.embedding_key = get_embedding_key(feat_config)
            self.loader.adata.obsm[self.embedding_key] = self.embedding
            
        else:
            # Single method: original behavior
            self.embedding_key = get_embedding_key(feat_config)
            feat_config['params']['save_dir'] = self.save_dir
            ExtractorClass = self.load_class(feat_config['module'], feat_config['class'])
            extractor = ExtractorClass(feat_config)
            self.embedding = extractor.fit_transform(self.loader)
            self.loader.adata.obsm[self.embedding_key] = self.embedding
    
    @timing
    def save_embeddings(self):
        """
        Save embeddings to a lightweight h5ad cache file.
        
        Saves only obs.index (cell IDs) and obsm (embeddings) for easy reuse.
        """
        # Create embeddings directory if needed
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lightweight adata with just cell indices and embeddings
        emb_adata = ad.AnnData(obs=self.loader.adata.obs[[]])  # Empty obs, keeps index
        emb_adata.obsm[self.embedding_key] = self.embedding
        emb_adata.uns['params'] = {
            'method': self.method_name,
            'dataset': self.dataset_name,
            'embedding_key': self.embedding_key,
            'n_cells': self.embedding.shape[0],
            'embedding_dim': self.embedding.shape[1],
        }
        
        # Save to cache
        emb_adata.write_h5ad(self.embedding_cache_path)
        self.log.info(f'Saved embeddings to: {self.embedding_cache_path}')

    def visualize_embedding(self):
        """
        Generate and save embedding visualizations using EmbeddingVisualizer.
        Also computes and saves PCA and UMAP to the original dataset (with caching).
        """
        visualizer = EmbeddingVisualizer(
            self.embedding, 
            self.loader.adata.obs, 
            save_dir=self.plots_embeddings_dir
        )
        visualizer.plot(
            adata_original=self.loader.adata,
            embedding_key=self.embedding_key,
            save_to_adata=True,
            pca_umap_cache_path=self.pca_umap_cache_path
        )

        
    @timing
    def run_evaluations(self):
        """
        Run configured evaluations.
        
        Evaluations are configured in the 'evaluations' section of the config.
        Supported evaluation types:
        - 'biological_signal': Cell type preservation and clustering quality
        - 'batch_effects': Batch correction quality
        - 'classification': Downstream classification task performance
        - 'annotation': KNN-based cell type annotation (label transfer from reference to query)
        
        Each evaluation should have:
        - type: evaluation type
        - params: parameters for the evaluation
        - skip: optional, if True, skip this evaluation
        """
        if not self.evaluations_config:
            self.log.info('No evaluations configured. Skipping evaluation step.')
            return
        
        self.log.info('=' * 60)
        self.log.info('Running configured evaluations')
        self.log.info('=' * 60)
        
        # Initialize evaluation runner with organized directories
        eval_runner = EvaluationRunner(
            adata=self.loader.adata,
            embedding_key=self.embedding_key,
            data_config=self.data_config,
            save_dir=self.save_dir,
            log=self.log,
            metrics_batch_effects_dir=self.metrics_batch_effects_dir,
            metrics_biological_signal_dir=self.metrics_biological_signal_dir,
            metrics_classification_dir=self.metrics_classification_dir,
            metrics_annotation_dir=self.metrics_annotation_dir,
            plots_evaluations_dir=self.plots_evaluations_dir
        )
        
        for eval_config in self.evaluations_config:
            if eval_config.get('skip', False):
                self.log.info(f"Skipping evaluation: {eval_config.get('type', 'unknown')}")
                continue
            
            eval_type = eval_config.get('type')
            if not eval_type:
                self.log.warning(f"Evaluation config missing 'type', skipping: {eval_config}")
                continue
            
            self.log.info(f"Running evaluation: {eval_type}")
            
            try:
                if eval_type == 'batch_effects':
                    eval_runner.run_batch_effects_evaluation(eval_config)
                elif eval_type == 'biological_signal':
                    eval_runner.run_biological_signal_evaluation(eval_config)
                elif eval_type == 'classification':
                    eval_runner.run_classification_evaluation(eval_config, self.load_class, self.loader)
                elif eval_type == 'annotation':
                    eval_runner.run_annotation_evaluation(eval_config)
                else:
                    self.log.warning(f"Unknown evaluation type: {eval_type}. Skipping.")
            except Exception as e:
                self.log.error(f"Error running evaluation {eval_type}: {e}", exc_info=True)
    

    
    def generate_summaries(self):
        """
        Generate summary comparison tables at multiple levels of aggregation.
        
        Creates scoped summaries at each level:
        - Method-level: Only datasets for this specific method within this task
        - Task-level: All methods and datasets for this specific task
        - Cross-task: All tasks, methods, and datasets (global view)
        
        Structure: OUTPUT_PATH / task / method / dataset /
        """
        try:
            from utils.results_summarizer import ResultsSummarizer
            
            task_name = self.task_name
            method_name = self.method_name
            
            # Generate summaries for this method within this task (scoped to method directory)
            # Only includes datasets for this specific method
            method_dir = OUTPUT_PATH / task_name / method_name
            method_summaries_dir = method_dir / 'summaries'
            summarizer = ResultsSummarizer(method_dir)
            summarizer.generate_all_summaries(save_dir=method_summaries_dir)
            self.log.info(f"Generated method-level summaries for {method_name} on task {task_name}")
            
            # Generate summaries for this task (scoped to task directory)
            # Includes all methods and datasets for this specific task
            task_dir = OUTPUT_PATH / task_name
            task_summaries_dir = task_dir / 'summaries'
            summarizer = ResultsSummarizer(task_dir)
            summarizer.generate_all_summaries(save_dir=task_summaries_dir)
            self.log.info(f"Generated task-level summaries for task: {task_name}")
            
            # Generate cross-task summaries (global scope)
            # Includes all tasks, methods, and datasets
            cross_task_summaries_dir = OUTPUT_PATH / 'summaries'
            summarizer = ResultsSummarizer(OUTPUT_PATH)
            summarizer.generate_all_summaries(save_dir=cross_task_summaries_dir)
            self.log.info("Generated cross-task summaries (all methods, datasets, tasks)")
            
        except Exception as e:
            self.log.warning(f"Error generating summaries: {e}. Continuing...")
            import traceback
            self.log.warning(traceback.format_exc())
            # Don't fail the experiment if summary generation fails


def main():
    """
    Main entry point for running an experiment from the command line.
    Sets random seed, parses config path, runs experiment, and saves timing log.
    """
    from run.utils import set_random_seed, _get_timing_log
    
    set_random_seed(42)
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'experiment.yaml'
    experiment = Experiment(config_path)
    experiment.run()
    timing_df = pd.DataFrame(_get_timing_log())
    timing_df.to_csv(join(experiment.save_dir, 'timing.csv'))

if __name__ == '__main__':
    main()