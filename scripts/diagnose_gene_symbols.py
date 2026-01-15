"""
Diagnostic script to check gene symbol conversion and matching with SCimilarity's gene_order.

This script helps debug why gene overlap might be low by:
1. Loading the dataset
2. Showing original var.index and feature_name
3. Converting to gene symbols
4. Loading SCimilarity's gene_order
5. Comparing and showing which genes match/don't match
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import anndata as ad
from data.data_loader import H5ADLoader
from scimilarity.cell_embedding import CellEmbedding


def normalize_symbol(symbol: str) -> str:
    """Normalize gene symbol: uppercase and strip whitespace."""
    return str(symbol).strip().upper()


def diagnose_gene_symbols(
    dataset_path: str,
    model_path: str,
    label_key: str = 'cell_type',
    batch_key: str = 'batch',
    layer_name: str = 'X',
    load_raw: bool = False
):
    """
    Diagnose gene symbol conversion and matching.
    
    Args:
        dataset_path: Path to H5AD file
        model_path: Path to SCimilarity model directory
        label_key: Key for cell type labels
        batch_key: Key for batch information
        layer_name: Layer to use (default: 'X')
        load_raw: Whether to load raw data
    """
    print("=" * 80)
    print("GENE SYMBOL DIAGNOSTIC SCRIPT")
    print("=" * 80)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    loader_config = {
        'path': dataset_path,
        'load_raw': load_raw,
        'label_key': label_key,
        'batch_key': batch_key,
        'layer_name': layer_name,
        'train_test_split': None,
        'cv_splits': None
    }
    loader = H5ADLoader(loader_config)
    adata = loader.load()
    
    print(f"   Dataset shape: {adata.shape}")
    print(f"   var.columns: {list(adata.var.columns)}")
    
    # 2. Show original var.index
    print("\n2. Original var.index (first 20):")
    print(f"   {list(adata.var.index[:20])}")
    
    # 3. Check feature_name column
    if 'feature_name' in adata.var.columns:
        print("\n3. feature_name column (first 20):")
        print(f"   {list(adata.var['feature_name'].astype(str)[:20])}")
        
        # Show sample of values
        print("\n   Sample comparison (first 10):")
        for i in range(min(10, len(adata.var))):
            orig = str(adata.var.index[i])
            feature = str(adata.var['feature_name'].iloc[i])
            normalized = normalize_symbol(feature)
            print(f"   [{i}] var.index: '{orig}' | feature_name: '{feature}' | normalized: '{normalized}'")
    else:
        print("\n3. No 'feature_name' column found in var")
    
    # 4. Convert to gene symbols
    print("\n4. Converting to gene symbols...")
    loader.ensure_gene_symbols_in_var_index(normalize=True)
    adata = loader.adata
    
    print(f"   After conversion (first 20):")
    print(f"   {list(adata.var.index[:20])}")
    
    # 5. Load SCimilarity's gene_order
    print("\n5. Loading SCimilarity gene_order...")
    try:
        ce = CellEmbedding(model_path=model_path, use_gpu=False)
        gene_order = ce.gene_order
        print(f"   Loaded {len(gene_order)} genes from SCimilarity model")
        print(f"   First 20 genes (original): {gene_order[:20]}")
        
        # Check if gene_order is normalized
        sample_gene_order = gene_order[:100]
        is_uppercase = sum(1 for g in sample_gene_order if str(g).isupper()) > len(sample_gene_order) * 0.8
        print(f"   Gene order appears to be uppercase: {is_uppercase}")
        
        # Normalize gene_order to uppercase (matching our dataset normalization)
        gene_order_normalized = [str(g).strip().upper() for g in gene_order]
        print(f"   First 20 genes (normalized to uppercase): {gene_order_normalized[:20]}")
    except Exception as e:
        print(f"   ERROR loading gene_order: {e}")
        return
    
    # 6. Compare gene sets (with and without normalization)
    print("\n6. Comparing gene sets...")
    dataset_genes = set(adata.var.index.astype(str))
    target_genes_original = set(str(g) for g in gene_order)
    target_genes_normalized = set(gene_order_normalized)
    
    # Comparison with original gene_order
    matching_genes_original = dataset_genes & target_genes_original
    dataset_only_original = dataset_genes - target_genes_original
    
    # Comparison with normalized gene_order (this is what we'll use)
    matching_genes = dataset_genes & target_genes_normalized
    dataset_only = dataset_genes - target_genes_normalized
    target_only = target_genes_normalized - dataset_genes
    
    print(f"   Dataset genes: {len(dataset_genes)}")
    print(f"   Target genes (original): {len(target_genes_original)}")
    print(f"   Target genes (normalized): {len(target_genes_normalized)}")
    print(f"\n   BEFORE normalization (original gene_order):")
    print(f"     Matching genes: {len(matching_genes_original)}")
    print(f"     Dataset-only genes: {len(dataset_only_original)}")
    print(f"\n   AFTER normalization (uppercase gene_order):")
    print(f"     Matching genes: {len(matching_genes)}")
    print(f"     Dataset-only genes: {len(dataset_only)}")
    print(f"     Target-only genes: {len(target_only)}")
    
    # Show improvement
    improvement = len(matching_genes) - len(matching_genes_original)
    if improvement > 0:
        print(f"\n   ✓ IMPROVEMENT: +{improvement} additional genes match after normalization!")
        print(f"     (This fixes case sensitivity issues)")
    
    # 7. Show samples of non-matching genes
    print("\n7. Sample of dataset genes that DON'T match target (first 20, using normalized gene_order):")
    print("   NOTE: These are genes in the dataset but NOT in SCimilarity's gene_order.")
    print("   This is NORMAL and EXPECTED - not all dataset genes are in SCimilarity's training set.")
    print("   Possible reasons:")
    print("     - Newer gene annotations not in SCimilarity's training data")
    print("     - Pseudogenes or gene types excluded from SCimilarity")
    print("     - Dataset-specific annotations")
    print("     - Genes filtered during SCimilarity's training")
    print()
    dataset_only_list = sorted(list(dataset_only))[:20]
    for gene in dataset_only_list:
        # Check if there's a case-insensitive match with original (should all be False now since we normalize)
        gene_upper = gene.upper()
        case_match_orig = any(g.upper() == gene_upper for g in target_genes_original)
        if case_match_orig:
            print(f"   '{gene}' (⚠️  would match original if case ignored - this should be fixed by normalization)")
        else:
            print(f"   '{gene}' (not in SCimilarity's gene_order - this is expected)")
    
    print("\n8. Sample of matching genes (first 20, using normalized gene_order):")
    matching_list = sorted(list(matching_genes))[:20]
    for gene in matching_list:
        print(f"   '{gene}'")
    
    # 9. Check for case sensitivity issues (using normalized gene_order)
    print("\n9. Checking for case sensitivity issues...")
    # Since we're now using normalized gene_order, case issues should be resolved
    # But let's check the original gene_order for comparison
    dataset_genes_upper = {g.upper() for g in dataset_genes}
    target_genes_original_upper = {g.upper() for g in target_genes_original}
    case_insensitive_matches_original = len(dataset_genes_upper & target_genes_original_upper)
    
    print(f"   Case-insensitive matches (with original gene_order): {case_insensitive_matches_original}")
    print(f"   Exact matches (with normalized gene_order): {len(matching_genes)}")
    
    if case_insensitive_matches_original > len(matching_genes_original):
        fixed_by_normalization = len(matching_genes) - len(matching_genes_original)
        print(f"   ✓ Fixed by normalization: {fixed_by_normalization} genes now match!")
        
        # Find genes that match case-insensitively but not exactly (with original)
        case_only_matches = (dataset_genes_upper & target_genes_original_upper) - {g.upper() for g in matching_genes_original}
        print(f"   Sample case-only matches that are now fixed (first 10):")
        count = 0
        for gene_upper in list(case_only_matches)[:10]:
            dataset_match = [g for g in dataset_genes if g.upper() == gene_upper]
            target_match_orig = [g for g in target_genes_original if g.upper() == gene_upper]
            if dataset_match and target_match_orig:
                print(f"     '{dataset_match[0]}' (dataset) vs '{target_match_orig[0]}' (target, original) → Now matches!")
                count += 1
                if count >= 10:
                    break
    
    # 10. Check for duplicate handling issues
    print("\n10. Checking for duplicate gene symbols...")
    dataset_genes_list = list(adata.var.index.astype(str))
    duplicates = [g for g in dataset_genes_list if dataset_genes_list.count(g) > 1]
    if duplicates:
        unique_duplicates = set(duplicates)
        print(f"   Found {len(unique_duplicates)} duplicate gene symbols in dataset")
        print(f"   Sample duplicates: {list(unique_duplicates)[:10]}")
    else:
        print("   No duplicates found")
    
    # 11. Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print(f"Gene overlap (with normalized gene_order): {len(matching_genes)}/{len(dataset_genes)} ({len(matching_genes)/len(dataset_genes)*100:.1f}%)")
    print(f"Gene overlap (with original gene_order): {len(matching_genes_original)}/{len(dataset_genes)} ({len(matching_genes_original)/len(dataset_genes)*100:.1f}%)")
    if improvement > 0:
        print(f"Improvement from normalization: +{improvement} genes ({improvement/len(dataset_genes)*100:.2f}%)")
    print(f"Required threshold: 5000 genes")
    
    if len(matching_genes) < 5000:
        print(f"\n⚠️  WARNING: Gene overlap ({len(matching_genes)}) is below threshold (5000)")
        if len(dataset_genes) < 5000:
            print(f"   → Dataset has fewer genes ({len(dataset_genes)}) than required threshold.")
            print(f"   → This is a dataset limitation, not a conversion issue.")
        else:
            print(f"   → Dataset has enough genes but many don't match target gene set.")
            print(f"   → This could be due to:")
            print(f"     - Different gene annotation versions")
            print(f"     - Different species")
            print(f"     - Dataset-specific filtering")
            print(f"     - HVG filtering reducing gene count (check if HVG is applied)")
    else:
        print(f"\n✓ Gene overlap is sufficient for alignment")
        if improvement > 0:
            print(f"   ✓ Case normalization fixed {improvement} additional gene matches!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose gene symbol conversion and matching with SCimilarity"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to H5AD dataset file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to SCimilarity model directory"
    )
    parser.add_argument(
        "--label_key",
        default="cell_type",
        help="Key for cell type labels (default: cell_type)"
    )
    parser.add_argument(
        "--batch_key",
        default="batch",
        help="Key for batch information (default: batch)"
    )
    parser.add_argument(
        "--layer",
        default="X",
        help="Layer to use (default: X)"
    )
    parser.add_argument(
        "--load_raw",
        action="store_true",
        help="Load raw data instead of processed"
    )
    
    args = parser.parse_args()
    
    diagnose_gene_symbols(
        dataset_path=args.dataset,
        model_path=args.model,
        label_key=args.label_key,
        batch_key=args.batch_key,
        layer_name=args.layer,
        load_raw=args.load_raw
    )
