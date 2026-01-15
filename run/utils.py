"""
Utility functions for experiment runner.
"""
import yaml
import time
from functools import wraps
import random
import numpy as np
import torch
from typing import Tuple, Dict, Any


# List to store timing records
def _get_timing_log():
    """Return the global timing log list."""
    global _timing_log
    if '_timing_log' not in globals():
        _timing_log = []
    return _timing_log

_timing_log = _get_timing_log()


def timing(func):
    """
    Decorator to measure and store execution time of a function.
    Appends timing info to the global _timing_log list.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        _timing_log.append({
            'function': func.__name__,
            'time_seconds': elapsed
        })
        return res
    return wrapper


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, sets PyTorch to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_configs(config_path: str) -> Tuple[Any, ...]:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Tuple containing run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, evaluations_config.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    run_id = config.get('run_id', None)
    data_config = config['dataset']
    qc_config = config['qc']
    preproc_config = config['preprocessing']
    feat_config = config['embedding']
    hvg_config = None
    if 'hvg' in config:
        hvg_config = config['hvg']
    
    # Handle new evaluations structure
    evaluations_config = config.get('evaluations', [])
    
    # Backward compatibility: convert old 'classification' section to new 'evaluations' format
    if 'classification' in config and not evaluations_config:
        classification_config = config['classification']
        # Convert old format to new format
        if not classification_config.get('skip', False):
            evaluations_config.append({
                'type': 'classification',
                'skip': False,
                'params': classification_config.get('params', {})
            })
    
    return run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, evaluations_config


def get_embedding_key(feat_config: Dict[str, Any]) -> str:
    """
    Compute the expected embedding key from the feature config.
    
    Args:
        feat_config: Feature/embedding configuration dictionary.
    
    Returns:
        str: The embedding key (e.g., 'X_scimilarity' for single method,
             'X_concatenate_scconcept_scimilarity' for multiple methods with concatenate,
             'X_avg_scconcept_scimilarity' for multiple methods with average, etc.).
    """
    if 'methods' in feat_config and isinstance(feat_config['methods'], list):
        # Multiple methods: key depends on joining method
        joining_method = feat_config.get('embedding_joining_method', 'concatenate')
        method_names = []
        for method_config in feat_config['methods']:
            method_name = method_config['name']
            method_names.append(method_name.lower())
        return f'X_{joining_method}_' + '_'.join(method_names)
    else:
        # Single method
        method_name = feat_config['method']
        return f'X_{method_name.lower()}'
