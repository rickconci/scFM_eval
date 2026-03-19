#!/usr/bin/env python
"""Standalone script to run Omnicell embedding in a subprocess.

Run with PYTHONPATH set so cell_types is importable, e.g.:
  PYTHONPATH=<repo_root>:<package_dir> OMNICELL_DATA_DIR=<...> \\
  python run_omnicell_embed.py --input in.h5ad --output out.npy \\
  --checkpoint_path ... --base_dir ... --config obs --gene_types feature_name --batch_size 4096

Expects adata.var to already have gene identifiers in var.index (e.g. gene_symbol)
and normalized names (no "SYMBOL (ID)"); the parent process prepares adata before writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Omnicell embedding (subprocess entrypoint)")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--checkpoint_path", required=True, help="Omnicell checkpoint .pt")
    parser.add_argument("--base_dir", required=True, help="cell_types package dir (cell-types/cell_types)")
    parser.add_argument("--config", default="obs", help="Cell-types config name")
    parser.add_argument("--gene_types", default="feature_name", choices=["feature_name", "feature_id"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--load_avg", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if (base / "cell_types").is_dir():
        repo_root = base
        package_dir = base / "cell_types"
    elif base.name == "cell_types":
        repo_root = base.parent
        package_dir = base
    else:
        repo_root = base.parent
        package_dir = base

    # Ensure cell_types and hydra_utils (under bulk_prediction) are importable
    path_add = [str(repo_root), str(package_dir)]
    if (package_dir / "bulk_prediction").is_dir():
        path_add.append(str(package_dir / "bulk_prediction"))
    for p in path_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    from cell_types.tasks.base import OmnicellBase
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = OmnicellBase(
        config=args.config,
        checkpoint=args.checkpoint_path,
        pert_config=None,
        pert_checkpoint=None,
        device=device,
        load_avg=args.load_avg,
        base_dir=str(package_dir),
    )

    adata = ad.read_h5ad(args.input)
    if hasattr(adata, "filename") and adata.isbacked:
        adata = adata.to_memory()
    adata = adata.copy()

    used_genes = getattr(base.generator.model, "used_genes", None)
    if used_genes is None:
        print("Omnicell generator has no used_genes", file=sys.stderr)
        return 1

    adata_mapped = base.map_genes(adata, args.gene_types, used_genes=used_genes)
    X = adata_mapped.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    embeddings = base.get_cell_embeddings(X, batch_size=args.batch_size)
    np.save(args.output, embeddings.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
