"""
KNN-based cell type annotation evaluation module.

Evaluates how well embeddings can be used to transfer cell type labels from a reference
dataset to a query dataset using k-nearest neighbors (KNN).

Supports multiple splitting strategies:
1. Even split: 50/50 per cell type
2. Random splits: Multiple random train/test splits
3. Difficult splits: Rare cell types skewed/absent in one split
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import scanpy as sc
from utils.logs_ import get_logger
from evaluation.eval import eval_classifier

logger = get_logger()


class AnnotationEvaluator:
    """
    Evaluates KNN-based cell type annotation performance.
    
    Splits data into reference (labeled) and query (unlabeled) sets, then uses
    KNN to transfer labels from reference to query.
    """
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        label_key: str = 'cell_type',
        save_dir: str = None,
        k: int = 15,
        metric: str = 'cosine',
        method: Literal['sklearn_knn', 'scanpy_ingest'] = 'sklearn_knn',
        split_strategies: List[Literal['even', 'random', 'difficult']] = ['even', 'random'],
        n_random_splits: int = 10,
        random_state: int = 42,
        min_cells_per_type: int = 2,
        **kwargs
    ):
        """
        Initialize annotation evaluator.
        
        Args:
            adata: AnnData object with embeddings and labels
            embedding_key: Key in adata.obsm containing embeddings
            label_key: Key in adata.obs containing cell type labels
            save_dir: Directory to save results
            k: Number of nearest neighbors for KNN
            metric: Distance metric ('cosine', 'euclidean', etc.)
            method: Annotation method ('sklearn_knn' or 'scanpy_ingest')
            split_strategies: List of splitting strategies to use
            n_random_splits: Number of random splits to generate
            random_state: Random seed for reproducibility
            min_cells_per_type: Minimum cells per type required for splitting
            **kwargs: Additional parameters
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.k = k
        self.metric = metric
        self.method = method
        self.split_strategies = split_strategies
        self.n_random_splits = n_random_splits
        self.random_state = random_state
        self.min_cells_per_type = min_cells_per_type
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate embedding exists
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")
        
        # Validate label key exists
        if label_key not in adata.obs:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs. Available: {list(adata.obs.columns)}")
    
    def evaluate(self) -> Dict[str, Dict]:
        """
        Run annotation evaluation with all specified splitting strategies.
        
        Returns:
            Dictionary mapping split strategy names to results dictionaries
        """
        logger.info(f'Evaluating annotation performance for embedding: {self.embedding_key}')
        logger.info(f'Using method: {self.method}, k={self.k}, metric={self.metric}')
        logger.info(f'Split strategies: {self.split_strategies}')
        
        all_results = {}
        
        # Get embeddings and labels
        X = self.adata.obsm[self.embedding_key]
        y = self.adata.obs[self.label_key].values
        
        # Get unique labels and their counts
        unique_labels, label_counts = np.unique(y, return_counts=True)
        label_names = unique_labels.tolist()
        logger.info(f'Found {len(label_names)} cell types: {label_names}')
        logger.info(f'Cell type counts: {dict(zip(label_names, label_counts))}')
        
        # Filter out cell types with too few cells
        valid_mask = label_counts >= self.min_cells_per_type
        valid_labels = unique_labels[valid_mask]
        valid_label_names = [lbl for lbl in label_names if lbl in valid_labels]
        
        if len(valid_label_names) < len(label_names):
            logger.warning(
                f'Filtering out {len(label_names) - len(valid_label_names)} cell types '
                f'with < {self.min_cells_per_type} cells. '
                f'Valid types: {valid_label_names}'
            )
            # Filter data to only valid labels
            valid_cell_mask = np.isin(y, valid_labels)
            X = X[valid_cell_mask]
            y = y[valid_cell_mask]
            label_names = valid_label_names
        
        # Run evaluation for each split strategy and save incrementally
        if 'even' in self.split_strategies:
            logger.info('=' * 60)
            logger.info('Running EVEN split strategy (50/50 per cell type)')
            logger.info('=' * 60)
            even_results = self._evaluate_even_split(X, y, label_names)
            all_results['even'] = even_results
            # Save immediately after each strategy
            self._save_strategy_results('even', even_results, label_names)
        
        if 'random' in self.split_strategies:
            logger.info('=' * 60)
            logger.info(f'Running RANDOM split strategy ({self.n_random_splits} splits)')
            logger.info('=' * 60)
            random_results = self._evaluate_random_splits(X, y, label_names)
            all_results['random'] = random_results
            # Save immediately after each strategy
            self._save_strategy_results('random', random_results, label_names)
        
        if 'difficult' in self.split_strategies:
            logger.info('=' * 60)
            logger.info('Running DIFFICULT split strategy (rare types skewed)')
            logger.info('=' * 60)
            difficult_results = self._evaluate_difficult_splits(X, y, label_names)
            all_results['difficult'] = difficult_results
            # Save immediately after each strategy
            self._save_strategy_results('difficult', difficult_results, label_names)
        
        # Create/update summary across all strategies
        self._save_summary(all_results, label_names)
        
        # Save to adata.uns
        if 'annotation' not in self.adata.uns:
            self.adata.uns['annotation'] = {}
        self.adata.uns['annotation'][self.embedding_key] = all_results
        logger.info(f"Saved annotation results to adata.uns['annotation']['{self.embedding_key}']")
        
        return all_results
    
    def _evaluate_even_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str]
    ) -> Dict:
        """Evaluate with even 50/50 split per cell type."""
        from sklearn.model_selection import train_test_split
        
        # Create label mapping to integers
        label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
        y_int = np.array([label_to_int[lbl] for lbl in y])
        
        # Split 50/50 per cell type
        X_ref, X_query, y_ref, y_query = train_test_split(
            X, y_int,
            test_size=0.5,
            stratify=y_int,
            random_state=self.random_state
        )
        
        logger.info(f'Even split: Reference={X_ref.shape[0]} cells, Query={X_query.shape[0]} cells')
        
        # Annotate query using reference
        y_pred, y_pred_score = self._annotate_with_knn(X_ref, y_ref, X_query, label_names)
        
        # Evaluate
        metrics_df, cls_report, per_class_df = eval_classifier(
            y_query, y_pred, y_pred_score, 'annotation_even', label_names
        )
        
        return {
            'metrics': metrics_df,
            'classification_report': cls_report,
            'per_class_metrics': per_class_df,
            'n_ref': X_ref.shape[0],
            'n_query': X_query.shape[0]
        }
    
    def _evaluate_random_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str]
    ) -> Dict:
        """Evaluate with multiple random splits."""
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Create label mapping to integers
        label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
        y_int = np.array([label_to_int[lbl] for lbl in y])
        
        # Generate multiple splits
        splitter = StratifiedShuffleSplit(
            n_splits=self.n_random_splits,
            test_size=0.5,
            random_state=self.random_state
        )
        
        all_metrics = []
        all_per_class_metrics = []
        
        for fold_idx, (ref_idx, query_idx) in enumerate(splitter.split(X, y_int)):
            logger.info(f'Random split {fold_idx + 1}/{self.n_random_splits}')
            
            X_ref = X[ref_idx]
            X_query = X[query_idx]
            y_ref = y_int[ref_idx]
            y_query = y_int[query_idx]
            
            # Annotate query using reference
            y_pred, y_pred_score = self._annotate_with_knn(X_ref, y_ref, X_query, label_names)
            
            # Evaluate
            metrics_df, cls_report, per_class_df = eval_classifier(
                y_query, y_pred, y_pred_score, f'annotation_random_fold{fold_idx}', label_names
            )
            
            all_metrics.append(metrics_df)
            all_per_class_metrics.append(per_class_df)
            
            # Save individual fold results immediately
            if self.save_dir:
                fold_metrics_file = self.save_dir / f'annotation_random_fold{fold_idx}_metrics.csv'
                metrics_df.to_csv(fold_metrics_file)
                fold_per_class_file = self.save_dir / f'annotation_random_fold{fold_idx}_per_class_metrics.csv'
                per_class_df.to_csv(fold_per_class_file)
                logger.info(f"Saved fold {fold_idx + 1} results to {fold_metrics_file}")
        
        # Aggregate across folds
        metrics_aggregated = self._aggregate_cv_metrics(all_metrics, 'annotation_random')
        per_class_aggregated = self._aggregate_cv_per_class_metrics(
            all_per_class_metrics, 'annotation_random'
        )
        
        return {
            'metrics': metrics_aggregated,
            'per_class_metrics': per_class_aggregated,
            'n_folds': self.n_random_splits,
            'n_ref_mean': X.shape[0] * 0.5,  # Approximate
            'n_query_mean': X.shape[0] * 0.5
        }
    
    def _evaluate_difficult_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str]
    ) -> Dict:
        """Evaluate with difficult splits where rare cell types are skewed."""
        from sklearn.model_selection import train_test_split
        
        # Create label mapping to integers
        label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
        y_int = np.array([label_to_int[lbl] for lbl in y])
        
        # Identify rare cell types (bottom 25% by count)
        unique_labels, label_counts = np.unique(y_int, return_counts=True)
        count_threshold = np.percentile(label_counts, 25)
        rare_labels = unique_labels[label_counts <= count_threshold]
        
        logger.info(f'Identified {len(rare_labels)} rare cell types: {[label_names[i] for i in rare_labels]}')
        
        # Create mask for rare vs common cells
        is_rare = np.isin(y_int, rare_labels)
        
        # Strategy: Put 80% of rare cells in query, 20% in reference
        # Put 50% of common cells in each
        ref_indices = []
        query_indices = []
        
        for label in unique_labels:
            label_mask = (y_int == label)
            label_indices = np.where(label_mask)[0]
            
            if label in rare_labels:
                # Rare: 20% ref, 80% query
                n_ref = max(1, int(len(label_indices) * 0.2))
                rng = np.random.default_rng(self.random_state)
                shuffled_indices = rng.permutation(label_indices)
                ref_indices.extend(shuffled_indices[:n_ref])
                query_indices.extend(shuffled_indices[n_ref:])
            else:
                # Common: 50/50 split
                n_ref = len(label_indices) // 2
                rng = np.random.default_rng(self.random_state)
                shuffled_indices = rng.permutation(label_indices)
                ref_indices.extend(shuffled_indices[:n_ref])
                query_indices.extend(shuffled_indices[n_ref:])
        
        ref_indices = np.array(ref_indices)
        query_indices = np.array(query_indices)
        
        X_ref = X[ref_indices]
        X_query = X[query_indices]
        y_ref = y_int[ref_indices]
        y_query = y_int[query_indices]
        
        logger.info(f'Difficult split: Reference={X_ref.shape[0]} cells, Query={X_query.shape[0]} cells')
        logger.info(f'Rare types in ref: {np.unique(y_ref[np.isin(y_ref, rare_labels)], return_counts=True)}')
        logger.info(f'Rare types in query: {np.unique(y_query[np.isin(y_query, rare_labels)], return_counts=True)}')
        
        # Annotate query using reference
        y_pred, y_pred_score = self._annotate_with_knn(X_ref, y_ref, X_query, label_names)
        
        # Evaluate
        metrics_df, cls_report, per_class_df = eval_classifier(
            y_query, y_pred, y_pred_score, 'annotation_difficult', label_names
        )
        
        return {
            'metrics': metrics_df,
            'classification_report': cls_report,
            'per_class_metrics': per_class_df,
            'n_ref': X_ref.shape[0],
            'n_query': X_query.shape[0],
            'rare_labels': [label_names[i] for i in rare_labels]
        }
    
    def _annotate_with_knn(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_query: np.ndarray,
        label_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Annotate query cells using KNN from reference.
        
        Args:
            X_ref: Reference embeddings (n_ref, n_features)
            y_ref: Reference labels as integers (n_ref,)
            X_query: Query embeddings (n_query, n_features)
            label_names: List of label names (for mapping)
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.method == 'sklearn_knn':
            # Use sklearn KNeighborsClassifier
            knn = KNeighborsClassifier(
                n_neighbors=self.k,
                metric=self.metric,
                weights='distance'  # Weight by distance
            )
            knn.fit(X_ref, y_ref)
            y_pred = knn.predict(X_query)
            y_pred_score = knn.predict_proba(X_query)
            
        elif self.method == 'scanpy_ingest':
            # Use scanpy ingest (requires AnnData objects)
            # Note: scanpy ingest works on PCA space, so we need to compute PCA first
            # Create temporary AnnData objects
            adata_ref = AnnData(X_ref)
            adata_ref.obs[self.label_key] = pd.Categorical([label_names[i] for i in y_ref])
            adata_query = AnnData(X_query)
            
            # Compute PCA on reference (scanpy ingest requires PCA)
            n_comps = min(50, min(X_ref.shape[0], X_ref.shape[1]) - 1)
            if n_comps < 1:
                n_comps = 1
            sc.tl.pca(adata_ref, n_comps=n_comps, random_state=self.random_state)
            
            # Ingest query onto reference PCA space
            sc.tl.ingest(adata_query, adata_ref, obs=self.label_key)
            
            # Get predictions
            y_pred_str = adata_query.obs[self.label_key].values
            label_to_int = {lbl: i for i, lbl in enumerate(label_names)}
            y_pred = np.array([label_to_int.get(lbl, 0) for lbl in y_pred_str])
            
            # For scanpy ingest, we don't get probabilities directly
            # Use a simple KNN to get probabilities for consistency
            knn = KNeighborsClassifier(
                n_neighbors=self.k,
                metric=self.metric,
                weights='distance'
            )
            knn.fit(X_ref, y_ref)
            y_pred_score = knn.predict_proba(X_query)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return y_pred, y_pred_score
    
    def _aggregate_cv_metrics(
        self,
        metrics_list: List[pd.DataFrame],
        estimator_name: str
    ) -> pd.DataFrame:
        """Aggregate metrics across CV folds (mean ± std)."""
        if not metrics_list:
            return pd.DataFrame()
        
        # Stack all metrics
        all_metrics = pd.concat(metrics_list, axis=1)
        
        # Compute mean and std
        mean_metrics = all_metrics.mean(axis=1)
        std_metrics = all_metrics.std(axis=1)
        
        # Create aggregated DataFrame
        aggregated = pd.DataFrame({
            'mean': mean_metrics,
            'std': std_metrics
        })
        aggregated.index.name = 'Metrics'
        
        # Also create a formatted version
        formatted = pd.DataFrame({
            estimator_name: [
                f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"
                for mean, std in zip(mean_metrics, std_metrics)
            ]
        }, index=mean_metrics.index)
        formatted.index.name = 'Metrics'
        
        return formatted
    
    def _aggregate_cv_per_class_metrics(
        self,
        per_class_list: List[pd.DataFrame],
        estimator_name: str
    ) -> pd.DataFrame:
        """Aggregate per-class metrics across CV folds (mean ± std)."""
        if not per_class_list:
            return pd.DataFrame()
        
        # Stack all per-class metrics
        all_per_class = pd.concat(per_class_list, axis=1)
        
        # Compute mean and std
        mean_per_class = all_per_class.mean(axis=1)
        std_per_class = all_per_class.std(axis=1)
        
        # Create formatted version
        formatted = pd.DataFrame({
            estimator_name: [
                f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"
                for mean, std in zip(mean_per_class, std_per_class)
            ]
        }, index=mean_per_class.index)
        formatted.index.name = 'Metrics'
        
        return formatted
    
    def _save_strategy_results(
        self,
        strategy_name: str,
        results: Dict,
        label_names: List[str]
    ) -> None:
        """
        Save results for a single strategy immediately.
        
        This ensures results are saved incrementally as each strategy completes.
        """
        if not self.save_dir:
            return
        
        # Save overall metrics
        if 'metrics' in results and results['metrics'] is not None:
            metrics_file = self.save_dir / f'annotation_{strategy_name}_metrics.csv'
            results['metrics'].to_csv(metrics_file)
            logger.info(f"Saved {strategy_name} metrics to {metrics_file}")
        
        # Save per-class metrics
        if 'per_class_metrics' in results and results['per_class_metrics'] is not None:
            per_class_file = self.save_dir / f'annotation_{strategy_name}_per_class_metrics.csv'
            results['per_class_metrics'].to_csv(per_class_file)
            logger.info(f"Saved {strategy_name} per-class metrics to {per_class_file}")
        
        # Save classification report if available
        if 'classification_report' in results and results['classification_report'] is not None:
            cls_report_file = self.save_dir / f'annotation_{strategy_name}_classification_report.csv'
            cls_report_df = pd.DataFrame(results['classification_report']).T
            cls_report_df.to_csv(cls_report_file)
            logger.info(f"Saved {strategy_name} classification report to {cls_report_file}")
        
        # Update summary file incrementally
        self._update_summary_file(strategy_name, results)
    
    def _update_summary_file(self, strategy_name: str, results: Dict) -> None:
        """
        Update summary file incrementally with results from a single strategy.
        
        Reads existing summary (if any), adds/updates the new strategy, and saves.
        """
        if not self.save_dir:
            return
        
        summary_file = self.save_dir / 'annotation_summary.csv'
        
        # Extract metrics for this strategy
        if 'metrics' not in results or results['metrics'] is None:
            return
        
        metrics = results['metrics']
        if not isinstance(metrics, pd.DataFrame) or len(metrics) == 0:
            return
        
        # Extract mean values (handle formatted strings)
        row = {'strategy': strategy_name}
        for metric_name in metrics.index:
            value = metrics.loc[metric_name, metrics.columns[0]]
            if isinstance(value, str):
                # Extract mean from "mean ± std" format
                try:
                    mean_val = float(value.split(' ± ')[0])
                except:
                    mean_val = np.nan
            else:
                mean_val = value
            row[metric_name] = mean_val
        
        # Read existing summary or create new
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                # Remove existing row for this strategy if it exists
                summary_df = summary_df[summary_df['strategy'] != strategy_name]
            except Exception as e:
                logger.warning(f"Error reading existing summary file: {e}. Creating new summary.")
                summary_df = pd.DataFrame()
        else:
            summary_df = pd.DataFrame()
        
        # Add new row
        new_row_df = pd.DataFrame([row])
        if len(summary_df) == 0:
            summary_df = new_row_df
        else:
            summary_df = pd.concat([summary_df, new_row_df], ignore_index=True)
        
        # Save updated summary
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Updated summary file with {strategy_name} results: {summary_file}")
    
    def _save_summary(self, all_results: Dict, label_names: List[str]) -> None:
        """
        Create/update summary file that aggregates metrics across all strategies.
        
        This is called after all strategies complete to ensure final summary is correct.
        Note: Summary is already updated incrementally, but this ensures consistency.
        """
        if not self.save_dir:
            return
        
        summary_rows = []
        for strategy_name, results in all_results.items():
            if 'metrics' in results and results['metrics'] is not None:
                metrics = results['metrics']
                if isinstance(metrics, pd.DataFrame) and len(metrics) > 0:
                    # Extract mean values (handle formatted strings)
                    row = {'strategy': strategy_name}
                    for metric_name in metrics.index:
                        value = metrics.loc[metric_name, metrics.columns[0]]
                        if isinstance(value, str):
                            # Extract mean from "mean ± std" format
                            try:
                                mean_val = float(value.split(' ± ')[0])
                            except:
                                mean_val = np.nan
                        else:
                            mean_val = value
                        row[metric_name] = mean_val
                    summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = self.save_dir / 'annotation_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Final summary saved to {summary_file}")
