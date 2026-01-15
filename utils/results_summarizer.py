"""
Results summarizer module.

Generates comprehensive comparison tables across methods, tasks, datasets, and metrics.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from utils.logs_ import get_logger

logger = get_logger()


class ResultsSummarizer:
    """
    Generates comparison tables and summaries across methods, tasks, datasets, and metrics.
    
    Supports different scoping levels:
    - 'root': Scans OUTPUT_PATH / task / method / dataset (full hierarchy)
    - 'task': Scans task_dir / method / dataset (single task)
    - 'method': Scans method_dir / dataset (single method within a task)
    """
    
    def __init__(self, output_base_dir: str, scope: str = 'auto'):
        """
        Initialize results summarizer.
        
        Args:
            output_base_dir: Base output directory to scan from.
            scope: Scanning scope - 'root', 'task', 'method', or 'auto' (detect automatically).
                   - 'root': Expects task/method/dataset structure
                   - 'task': Expects method/dataset structure (task_name inferred from dir name)
                   - 'method': Expects dataset structure (task/method inferred from path)
                   - 'auto': Automatically detect based on directory structure
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine scope
        if scope == 'auto':
            self.scope = self._detect_scope()
        else:
            self.scope = scope
    
    def _detect_scope(self) -> str:
        """
        Auto-detect the scanning scope based on directory structure.
        
        Looks for 'metrics' directories at different depths to determine the scope:
        - If metrics found at depth 3 (task/method/dataset/metrics): 'root' scope
        - If metrics found at depth 2 (method/dataset/metrics): 'task' scope  
        - If metrics found at depth 1 (dataset/metrics): 'method' scope
        
        Returns:
            Detected scope: 'root', 'task', or 'method'
        """
        # Check for metrics at different depths
        # Depth 1: method scope (dataset/metrics)
        for child in self.output_base_dir.iterdir():
            if child.is_dir() and not child.name.startswith('.') and child.name not in ['summaries', 'embeddings']:
                metrics_dir = child / 'metrics'
                if metrics_dir.exists():
                    logger.info(f"Detected 'method' scope (dataset/metrics found)")
                    return 'method'
        
        # Depth 2: task scope (method/dataset/metrics)
        for child in self.output_base_dir.iterdir():
            if child.is_dir() and not child.name.startswith('.') and child.name not in ['summaries', 'embeddings']:
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and not grandchild.name.startswith('.'):
                        metrics_dir = grandchild / 'metrics'
                        if metrics_dir.exists():
                            logger.info(f"Detected 'task' scope (method/dataset/metrics found)")
                            return 'task'
        
        # Default to root scope
        logger.info(f"Detected 'root' scope (default)")
        return 'root'
    
    def collect_metrics_from_experiment(
        self, 
        experiment_dir: Path,
        dataset_name: str,
        task_name: str,
        method_name: str
    ) -> Dict[str, float]:
        """
        Collect all metrics from a single experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory (e.g., __output/dataset/task/method/)
            dataset_name: Name of the dataset
            task_name: Name of the task
            method_name: Name of the method
            
        Returns:
            Dictionary of metric_name: value
        """
        metrics_dict = {}
        
        # Determine the main embedding key from method_name
        # e.g., method_name='gtex_v9_scimilarity' -> embedding_key='X_scimilarity'
        main_embedding_key = f'X_{method_name.split("_")[-1]}'
        
        # Collect batch effects metrics
        batch_effects_file = experiment_dir / 'metrics' / 'batch_effects' / 'batch_effects_metrics.csv'
        if batch_effects_file.exists():
            try:
                df = pd.read_csv(batch_effects_file, index_col=0)
                # Find the main embedding row (not baseline)
                main_rows = [idx for idx in df.index if not idx.startswith('X_baseline_')]
                if main_rows:
                    row_idx = main_rows[0]
                    for col in df.columns:
                        metrics_dict[f'batch_effects_{col}'] = df.loc[row_idx, col]
                elif df.shape[0] > 0:
                    # Fallback: use first row
                    for col in df.columns:
                        metrics_dict[f'batch_effects_{col}'] = df[col].iloc[0]
            except Exception as e:
                logger.warning(f"Error reading batch_effects metrics from {batch_effects_file}: {e}")
        
        # Collect biological signal metrics
        bio_signal_file = experiment_dir / 'metrics' / 'biological_signal' / 'biological_signal_metrics.csv'
        if bio_signal_file.exists():
            try:
                df = pd.read_csv(bio_signal_file, index_col=0)
                # Find the main embedding row (not baseline)
                main_rows = [idx for idx in df.index if not idx.startswith('X_baseline_')]
                if main_rows:
                    row_idx = main_rows[0]
                    for col in df.columns:
                        metrics_dict[f'biological_signal_{col}'] = df.loc[row_idx, col]
                elif df.shape[0] > 0:
                    # Fallback: use first row
                    for col in df.columns:
                        metrics_dict[f'biological_signal_{col}'] = df[col].iloc[0]
            except Exception as e:
                logger.warning(f"Error reading biological_signal metrics from {bio_signal_file}: {e}")
        
        # Collect classification metrics (may have multiple files)
        classification_dir = experiment_dir / 'metrics' / 'classification'
        if classification_dir.exists():
            # Priority 1: CV aggregated metrics (mean across folds)
            for csv_file in classification_dir.glob('*_cv_metrics_aggregated.csv'):
                if 'train' not in csv_file.name and 'per_class' not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        # Extract train function from filename (e.g., randomforest_avg_cv_metrics_aggregated.csv)
                        train_func = csv_file.stem.replace('_cv_metrics_aggregated', '')
                        for metric_name in df.index:
                            if 'mean' in df.columns:
                                value = df.loc[metric_name, 'mean']
                                metrics_dict[f'classification_{train_func}_{metric_name}'] = value
                    except Exception as e:
                        logger.warning(f"Error reading CV aggregated metrics from {csv_file}: {e}")
            
            # Priority 2: Non-CV metrics (cls_metrics_*.csv)
            for csv_file in classification_dir.glob('cls_metrics_*.csv'):
                if 'train' not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file, index_col=0)
                        # Extract postfix from filename (e.g., cls_metrics_avg_expr.csv -> avg_expr)
                        postfix = csv_file.stem.replace('cls_metrics_', '')
                        for metric_name in df.index:
                            if len(df.columns) > 0:
                                value = df.iloc[df.index.get_loc(metric_name), 0]
                                # Only add if not already from CV aggregated
                                key = f'classification_{postfix}_{metric_name}'
                                if key not in metrics_dict:
                                    metrics_dict[key] = value
                    except Exception as e:
                        logger.warning(f"Error reading cls_metrics from {csv_file}: {e}")
            
            # Priority 3: Other metrics files (fallback)
            for csv_file in classification_dir.glob('*_metrics.csv'):
                if 'cv' not in csv_file.name and 'cls_metrics' not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file)
                        # Handle different formats
                        if 'Metrics' in df.columns:
                            # Long format: pivot
                            metric_col = [c for c in df.columns if c != 'Metrics'][0]
                            for _, row in df.iterrows():
                                metric_name = row['Metrics']
                                value = row[metric_col]
                                method_suffix = csv_file.stem.replace('_metrics', '')
                                key = f'classification_{method_suffix}_{metric_name}'
                                if key not in metrics_dict:
                                    metrics_dict[key] = value
                        else:
                            # Wide format: use columns as metrics
                            for col in df.columns:
                                if col not in ['Unnamed: 0', 'index']:
                                    method_suffix = csv_file.stem.replace('_metrics', '')
                                    key = f'classification_{method_suffix}_{col}'
                                    if key not in metrics_dict:
                                        metrics_dict[key] = df[col].iloc[0]
                    except Exception as e:
                        logger.warning(f"Error reading classification metrics from {csv_file}: {e}")
        
        # Collect annotation metrics
        annotation_dir = experiment_dir / 'metrics' / 'annotation'
        if annotation_dir.exists():
            # Read summary file if it exists
            summary_file = annotation_dir / 'annotation_summary.csv'
            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    # Each row is a strategy (even, random, difficult)
                    for _, row in df.iterrows():
                        strategy = row.get('strategy', 'unknown')
                        for col in df.columns:
                            if col != 'strategy':
                                try:
                                    value = float(row[col])
                                    metrics_dict[f'annotation_{strategy}_{col}'] = value
                                except (ValueError, TypeError):
                                    pass  # Skip non-numeric columns
                except Exception as e:
                    logger.warning(f"Error reading annotation summary from {summary_file}: {e}")
            else:
                # Read individual strategy files
                for csv_file in annotation_dir.glob('annotation_*_metrics.csv'):
                    if 'fold' not in csv_file.name and 'per_class' not in csv_file.name:
                        try:
                            df = pd.read_csv(csv_file, index_col=0)
                            # Extract strategy name from filename
                            strategy = csv_file.stem.replace('annotation_', '').replace('_metrics', '')
                            for metric_name in df.index:
                                if len(df.columns) > 0:
                                    value = df.iloc[df.index.get_loc(metric_name), 0]
                                    # Handle formatted strings like "0.95 ± 0.02"
                                    if isinstance(value, str) and '±' in value:
                                        try:
                                            value = float(value.split('±')[0].strip())
                                        except ValueError:
                                            continue
                                    metrics_dict[f'annotation_{strategy}_{metric_name}'] = value
                        except Exception as e:
                            logger.warning(f"Error reading annotation metrics from {csv_file}: {e}")
        
        return metrics_dict
    
    def scan_all_experiments(self) -> pd.DataFrame:
        """
        Scan experiment directories and collect metrics based on the detected scope.
        
        Scope determines the directory structure:
        - 'root': OUTPUT_PATH / task / method / dataset /
        - 'task': task_dir / method / dataset /
        - 'method': method_dir / dataset /
        
        Returns:
            DataFrame with columns: dataset, task, method, and all metric columns
        """
        all_results = []
        
        if self.scope == 'method':
            # Method scope: base_dir is method directory, scan datasets directly
            # Infer task_name and method_name from directory path
            method_name = self.output_base_dir.name
            task_name = self.output_base_dir.parent.name
            
            for dataset_dir in self.output_base_dir.iterdir():
                if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                    continue
                if dataset_dir.name in ['summaries', 'config', 'logs', 'plots', 'metrics']:
                    continue
                
                dataset_name = dataset_dir.name
                metrics = self.collect_metrics_from_experiment(
                    dataset_dir, dataset_name, task_name, method_name
                )
                
                if metrics:
                    result_row = {
                        'dataset': dataset_name,
                        'task': task_name,
                        'method': method_name,
                        **metrics
                    }
                    all_results.append(result_row)
        
        elif self.scope == 'task':
            # Task scope: base_dir is task directory, scan method/dataset
            task_name = self.output_base_dir.name
            
            for method_dir in self.output_base_dir.iterdir():
                if not method_dir.is_dir() or method_dir.name.startswith('.'):
                    continue
                if method_dir.name in ['summaries', 'embeddings']:
                    continue
                
                method_name = method_dir.name
                
                for dataset_dir in method_dir.iterdir():
                    if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                        continue
                    if dataset_dir.name in ['summaries', 'config', 'logs', 'plots', 'metrics']:
                        continue
                    
                    dataset_name = dataset_dir.name
                    metrics = self.collect_metrics_from_experiment(
                        dataset_dir, dataset_name, task_name, method_name
                    )
                    
                    if metrics:
                        result_row = {
                            'dataset': dataset_name,
                            'task': task_name,
                            'method': method_name,
                            **metrics
                        }
                        all_results.append(result_row)
        
        else:
            # Root scope: base_dir is OUTPUT_PATH, scan task/method/dataset
            for task_dir in self.output_base_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith('.'):
                    continue
                if task_dir.name in ['embeddings', 'summaries']:
                    continue
                
                task_name = task_dir.name
                
                for method_dir in task_dir.iterdir():
                    if not method_dir.is_dir() or method_dir.name.startswith('.'):
                        continue
                    if method_dir.name == 'summaries':
                        continue
                    
                    method_name = method_dir.name
                    
                    for dataset_dir in method_dir.iterdir():
                        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                            continue
                        if dataset_dir.name in ['summaries', 'config', 'logs', 'plots', 'metrics']:
                            continue
                        
                        dataset_name = dataset_dir.name
                        metrics = self.collect_metrics_from_experiment(
                            dataset_dir, dataset_name, task_name, method_name
                        )
                        
                        if metrics:
                            result_row = {
                                'dataset': dataset_name,
                                'task': task_name,
                                'method': method_name,
                                **metrics
                            }
                            all_results.append(result_row)
        
        if not all_results:
            logger.warning("No experiment results found!")
            return pd.DataFrame()
        
        return pd.DataFrame(all_results)
    
    def create_comprehensive_comparison_table(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive comparison table: Methods × (Tasks × Datasets × Metrics).
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include (if None, includes all)
            
        Returns:
            DataFrame with methods as rows, task×dataset×metric as columns
        """
        if results_df.empty:
            return pd.DataFrame()
        
        # Get all metric columns (exclude dataset, task, method)
        if metric_columns is None:
            metric_columns = [col for col in results_df.columns 
                            if col not in ['dataset', 'task', 'method']]
        
        # Create wide format: method × (task_dataset_metric)
        comparison_rows = []
        
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            # Create columns: task_dataset_metric
            for task in results_df['task'].unique():
                task_df = method_df[method_df['task'] == task]
                for dataset in results_df['dataset'].unique():
                    dataset_df = task_df[task_df['dataset'] == dataset]
                    if not dataset_df.empty:
                        for metric in metric_columns:
                            if metric in dataset_df.columns:
                                col_name = f'{task}_{dataset}_{metric}'
                                value = dataset_df[metric].iloc[0]
                                row[col_name] = value
                            else:
                                col_name = f'{task}_{dataset}_{metric}'
                                row[col_name] = np.nan
                    else:
                        # No data for this combination
                        for metric in metric_columns:
                            col_name = f'{task}_{dataset}_{metric}'
                            row[col_name] = np.nan
            
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def create_distribution_summary(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create distribution summary: Methods × Tasks with metric distributions across datasets.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include
            
        Returns:
            DataFrame with methods as rows, task_metric as columns showing distributions
        """
        if results_df.empty:
            return pd.DataFrame()
        
        if metric_columns is None:
            metric_columns = [col for col in results_df.columns 
                            if col not in ['dataset', 'task', 'method']]
        
        distribution_rows = []
        
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            for task in results_df['task'].unique():
                task_df = method_df[method_df['task'] == task]
                for metric in metric_columns:
                    if metric in task_df.columns:
                        values = task_df[metric].dropna()
                        if len(values) > 0:
                            col_name = f'{task}_{metric}'
                            # Store as string: mean±std (min, max) [n_datasets]
                            mean_val = values.mean()
                            std_val = values.std()
                            min_val = values.min()
                            max_val = values.max()
                            n_datasets = len(values)
                            row[col_name] = f'{mean_val:.4f}±{std_val:.4f} ({min_val:.4f}, {max_val:.4f}) [n={n_datasets}]'
                        else:
                            col_name = f'{task}_{metric}'
                            row[col_name] = 'N/A'
                    else:
                        col_name = f'{task}_{metric}'
                        row[col_name] = 'N/A'
            
            distribution_rows.append(row)
        
        return pd.DataFrame(distribution_rows)
    
    def create_task_aggregated_table(
        self,
        results_df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Create table with methods × tasks, aggregating across datasets.
        
        Args:
            results_df: DataFrame from scan_all_experiments()
            metric_columns: List of metric columns to include
            aggregation: Aggregation method ('mean', 'median', 'std', 'min', 'max')
            
        Returns:
            DataFrame with methods as rows, task_metric as columns
        """
        if results_df.empty:
            return pd.DataFrame()
        
        if metric_columns is None:
            metric_columns = [col for col in results_df.columns 
                            if col not in ['dataset', 'task', 'method']]
        
        aggregated_rows = []
        
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            row = {'method': method}
            
            for task in results_df['task'].unique():
                task_df = method_df[method_df['task'] == task]
                for metric in metric_columns:
                    if metric in task_df.columns:
                        values = task_df[metric].dropna()
                        if len(values) > 0:
                            if aggregation == 'mean':
                                value = values.mean()
                            elif aggregation == 'median':
                                value = values.median()
                            elif aggregation == 'std':
                                value = values.std()
                            elif aggregation == 'min':
                                value = values.min()
                            elif aggregation == 'max':
                                value = values.max()
                            else:
                                value = values.mean()
                            
                            col_name = f'{task}_{metric}'
                            row[col_name] = value
                        else:
                            col_name = f'{task}_{metric}'
                            row[col_name] = np.nan
                    else:
                        col_name = f'{task}_{metric}'
                        row[col_name] = np.nan
            
            aggregated_rows.append(row)
        
        return pd.DataFrame(aggregated_rows)
    
    def generate_all_summaries(self, save_dir: Optional[Path] = None):
        """
        Generate all summary tables and save them.
        
        Args:
            save_dir: Directory to save summaries (default: output_base_dir/summaries)
        """
        if save_dir is None:
            save_dir = self.output_base_dir / 'summaries'
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Scanning all experiment results...")
        results_df = self.scan_all_experiments()
        
        if results_df.empty:
            logger.warning("No results found. Cannot generate summaries.")
            return
        
        logger.info(f"Found results for {len(results_df)} experiment configurations")
        logger.info(f"Datasets: {results_df['dataset'].unique()}")
        logger.info(f"Tasks: {results_df['task'].unique()}")
        logger.info(f"Methods: {results_df['method'].unique()}")
        
        # Get metric columns
        metric_columns = [col for col in results_df.columns 
                         if col not in ['dataset', 'task', 'method']]
        
        # 1. Comprehensive comparison table (methods × task_dataset_metric)
        logger.info("Creating comprehensive comparison table...")
        comprehensive_table = self.create_comprehensive_comparison_table(
            results_df, metric_columns
        )
        if not comprehensive_table.empty:
            comprehensive_path = save_dir / 'comprehensive_comparison.csv'
            comprehensive_table.to_csv(comprehensive_path, index=False)
            logger.info(f"Saved comprehensive comparison to {comprehensive_path}")
        
        # 2. Distribution summary (methods × task_metric with distributions)
        logger.info("Creating distribution summary...")
        distribution_summary = self.create_distribution_summary(
            results_df, metric_columns
        )
        if not distribution_summary.empty:
            distribution_path = save_dir / 'distribution_summary.csv'
            distribution_summary.to_csv(distribution_path, index=False)
            logger.info(f"Saved distribution summary to {distribution_path}")
        
        # 3. Task-aggregated tables (mean, median, std across datasets)
        for agg_func in ['mean', 'median', 'std']:
            logger.info(f"Creating task-aggregated table ({agg_func})...")
            aggregated_table = self.create_task_aggregated_table(
                results_df, metric_columns, aggregation=agg_func
            )
            if not aggregated_table.empty:
                agg_path = save_dir / f'task_aggregated_{agg_func}.csv'
                aggregated_table.to_csv(agg_path, index=False)
                logger.info(f"Saved {agg_func} aggregated table to {agg_path}")
        
        # 4. Save raw collected results
        raw_path = save_dir / 'all_results_raw.csv'
        results_df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw results to {raw_path}")
        
        logger.info(f"All summaries saved to {save_dir}")
