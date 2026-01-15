"""
Evaluation runner module.

Handles running different types of evaluations (batch effects, biological signal, classification).
"""
from typing import Dict
from anndata import AnnData
from utils.logs_ import get_logger

logger = get_logger()


class EvaluationRunner:
    """
    Handles running configured evaluations.
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        data_config: Dict,
        save_dir: str,
        log,
        metrics_batch_effects_dir: str = None,
        metrics_biological_signal_dir: str = None,
        metrics_classification_dir: str = None,
        metrics_annotation_dir: str = None,
        plots_evaluations_dir: str = None
    ):
        """
        Initialize evaluation runner.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm containing embeddings
            data_config: Dataset configuration (for default keys)
            save_dir: Base directory to save results (for backward compatibility)
            log: Logger instance
            metrics_batch_effects_dir: Directory for batch effects metrics
            metrics_biological_signal_dir: Directory for biological signal metrics
            metrics_classification_dir: Directory for classification metrics
            metrics_annotation_dir: Directory for annotation metrics
            plots_evaluations_dir: Directory for evaluation plots
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.data_config = data_config
        self.save_dir = save_dir  # Keep for backward compatibility
        self.log = log
        
        # Use organized subdirectories if provided, otherwise fall back to save_dir
        self.metrics_batch_effects_dir = metrics_batch_effects_dir or save_dir
        self.metrics_biological_signal_dir = metrics_biological_signal_dir or save_dir
        self.metrics_classification_dir = metrics_classification_dir or save_dir
        self.metrics_annotation_dir = metrics_annotation_dir or save_dir
        self.plots_evaluations_dir = plots_evaluations_dir or save_dir
    
    def run_batch_effects_evaluation(self, eval_config: Dict) -> None:
        """Run batch effects evaluation."""
        from evaluation.batch_effects import BatchEffectsEvaluator
        
        params = eval_config.get('params', {})
        batch_key = params.get('batch_key', self.data_config.get('batch_key', 'batch'))
        label_key = params.get('label_key', self.data_config.get('label_key', 'label'))
        metric = params.get('metric', 'euclidean')
        run_baselines = params.get('run_baselines', False)
        baseline_methods = params.get('baseline_methods', [])
        
        # Evaluate main embedding
        evaluator = BatchEffectsEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            batch_key=batch_key,
            label_key=label_key,
            save_dir=self.metrics_batch_effects_dir,
            metric=metric,
            **{k: v for k, v in params.items() if k not in ['batch_key', 'label_key', 'metric', 'run_baselines', 'baseline_methods']}
        )
        results = evaluator.evaluate()
        self.log.info(f"Batch effects evaluation complete. Results: {results}")
        
        # Run baseline evaluations if requested
        if run_baselines:
            self._run_baseline_batch_effects_evaluations(
                eval_config, batch_key, label_key, metric, baseline_methods
            )
    
    def _run_baseline_batch_effects_evaluations(
        self,
        eval_config: Dict,
        batch_key: str,
        label_key: str,
        metric: str,
        baseline_methods: list
    ) -> None:
        """
        Run batch effects evaluation on baseline embeddings.
        
        Baseline evaluations are dataset-specific (not method-specific), so we:
        1. Check if baseline results already exist in the dataset
        2. Only run evaluations for baselines that don't exist yet
        3. Save results to adata.uns['baseline_batch_effects']
        """
        from evaluation.batch_effects import BatchEffectsEvaluator
        from evaluation.baseline_embeddings import create_baseline_embedding, BASELINE_METHODS
        
        params = eval_config.get('params', {})
        
        # Default baseline methods if not specified
        if not baseline_methods:
            baseline_methods = BASELINE_METHODS
        
        # Check which baselines already exist in the dataset
        existing_baselines = set()
        if 'baseline_batch_effects' in self.adata.uns:
            existing_baselines = set(self.adata.uns['baseline_batch_effects'].keys())
        
        # Also check if embeddings exist (they might have been computed but not evaluated)
        existing_embeddings = set()
        for key in self.adata.obsm.keys():
            if key.startswith('X_baseline_'):
                existing_embeddings.add(key)
        
        self.log.info(f'Checking baseline evaluations. Existing: {existing_baselines}')
        
        # Filter out baselines that already have results
        baselines_to_run = []
        for baseline_method in baseline_methods:
            if baseline_method not in BASELINE_METHODS:
                self.log.warning(f"Unknown baseline method: {baseline_method}. Skipping.")
                continue
            
            baseline_key = f'X_baseline_{baseline_method}'
            
            # Skip if results already exist
            if baseline_key in existing_baselines:
                self.log.info(f'Baseline {baseline_method} already evaluated. Skipping.')
                continue
            
            baselines_to_run.append(baseline_method)
        
        if not baselines_to_run:
            self.log.info('All requested baseline evaluations already exist in dataset. Skipping.')
            return
        
        self.log.info(f'Running baseline evaluations for methods: {baselines_to_run}')
        
        for baseline_method in baselines_to_run:
            baseline_key = f'X_baseline_{baseline_method}'
            
            try:
                # Check if embedding already exists, otherwise create it
                if baseline_key not in self.adata.obsm:
                    self.log.info(f'Creating baseline embedding: {baseline_method}')
                    baseline_embedding = create_baseline_embedding(
                        adata=self.adata,
                        method=baseline_method,
                        batch_key=batch_key,
                        label_key=label_key,
                        use_rep='X',
                        n_comps=50,
                        random_seed=42
                    )
                    # Store in obsm
                    self.adata.obsm[baseline_key] = baseline_embedding
                else:
                    self.log.info(f'Using existing baseline embedding: {baseline_method}')
                
                # Evaluate baseline
                self.log.info(f'Evaluating baseline: {baseline_method}')
                baseline_evaluator = BatchEffectsEvaluator(
                    adata=self.adata,
                    embedding_key=baseline_key,
                    batch_key=batch_key,
                    label_key=label_key,
                    save_dir=self.metrics_batch_effects_dir,
                    metric=metric,
                    **{k: v for k, v in params.items() if k not in ['batch_key', 'label_key', 'metric', 'run_baselines', 'baseline_methods']}
                )
                baseline_results = baseline_evaluator.evaluate()
                self.log.info(f"Baseline evaluation complete for {baseline_method}. Results: {baseline_results}")
                
            except Exception as e:
                self.log.error(f"Error evaluating baseline {baseline_method}: {e}", exc_info=True)
    
    def run_biological_signal_evaluation(self, eval_config: Dict) -> None:
        """Run biological signal evaluation."""
        from evaluation.biological_signal import BiologicalSignalEvaluator
        
        params = eval_config.get('params', {})
        label_key = params.get('label_key', self.data_config.get('label_key', 'label'))
        metric = params.get('metric', 'euclidean')
        
        evaluator = BiologicalSignalEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            label_key=label_key,
            save_dir=self.metrics_biological_signal_dir,
            metric=metric,
            **{k: v for k, v in params.items() if k not in ['label_key', 'metric']}
        )
        results = evaluator.evaluate()
        self.log.info(f"Biological signal evaluation complete. Results: {results}")
    
    def run_classification_evaluation(
        self, 
        eval_config: Dict, 
        load_class_func,
        loader
    ) -> None:
        """
        Run classification evaluation (train and evaluate classifier).
        
        Args:
            eval_config: Evaluation configuration dictionary
            load_class_func: Function to load classes dynamically (from Experiment.load_class)
            loader: Data loader object (required by ClassifierPipeline)
        """
        params = eval_config.get('params', {}).copy()  # Copy to avoid modifying original
        
        # Extract module, class, viz, eval from params
        module = params.pop('module', 'models.classify')
        class_name = params.pop('class', 'ClassifierPipeline')
        viz = params.pop('viz', False)
        eval_ = params.pop('eval', True)
        
        # Remaining params are classifier parameters
        clf_params = params.copy()
        clf_params['save_dir'] = self.metrics_classification_dir
        clf_params['plots_dir'] = self.plots_evaluations_dir  # Separate plots directory
        clf_params['embedding_col'] = self.embedding_key
        
        # Build classifier config
        clf_config = {
            'module': module,
            'class': class_name,
            'params': clf_params,
            'viz': viz,
            'eval': eval_,
        }
        
        # Train and evaluate classifier
        ClfClass = load_class_func(module, class_name)
        clf = ClfClass(clf_config)
        clf.train(loader)
        
        self.log.info(f"Classification evaluation complete for embedding: {self.embedding_key}")
    
    def run_annotation_evaluation(self, eval_config: Dict) -> None:
        """Run annotation evaluation (KNN-based label transfer)."""
        from evaluation.annotation import AnnotationEvaluator
        
        params = eval_config.get('params', {})
        label_key = params.get('label_key', self.data_config.get('label_key', 'cell_type'))
        k = params.get('k', 15)
        metric = params.get('metric', 'cosine')
        method = params.get('method', 'sklearn_knn')
        split_strategies = params.get('split_strategies', ['even', 'random'])
        n_random_splits = params.get('n_random_splits', 10)
        random_state = params.get('random_state', 42)
        min_cells_per_type = params.get('min_cells_per_type', 2)
        
        evaluator = AnnotationEvaluator(
            adata=self.adata,
            embedding_key=self.embedding_key,
            label_key=label_key,
            save_dir=self.metrics_annotation_dir,
            k=k,
            metric=metric,
            method=method,
            split_strategies=split_strategies,
            n_random_splits=n_random_splits,
            random_state=random_state,
            min_cells_per_type=min_cells_per_type,
            **{k: v for k, v in params.items() if k not in [
                'label_key', 'k', 'metric', 'method', 'split_strategies',
                'n_random_splits', 'random_state', 'min_cells_per_type'
            ]}
        )
        results = evaluator.evaluate()
        self.log.info(f"Annotation evaluation complete. Results summary: {list(results.keys())}")
