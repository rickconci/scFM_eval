#!/usr/bin/env python
"""Standalone script to run Omnicell embedding in a subprocess.

Run with PYTHONPATH set so cell_types is importable, e.g.:
  PYTHONPATH=<repo_root>:<package_dir> OMNICELL_DATA_DIR=<...> \\
  python run_omnicell_embed.py --input in.h5ad --output out.npy \\
  --checkpoint_path ... --base_dir ... --config obs --gene_types feature_name \\
  --encoding-method singleton --batch_size 4096
  (checkpoint loading uses averaged weights by default, like generate_distribution_embeddings.py;
  pass --no-load-avg for raw encoder_state_dict)

Expects adata.var to already have gene identifiers in var.index (e.g. gene_symbol)
and normalized names (no "SYMBOL (ID)"); the parent process prepares adata before writing.

**Encoding methods** (match ``scFM_eval/debug_encoder.py``):

- ``singleton`` *(default)* — per-cell, ``(B, 1, G)`` with ``return_cell_embeddings=False``.
  For ``S=1`` the encoder's set-level ``lat`` **is** the per-cell embedding, so no extra head.
  Chunked by ``--batch_size``. Matches current extractor behavior.
- ``joint`` — one forward over ``(1, N, G)`` with ``return_cell_embeddings=True``. SetNorm
  sees all N cells in one forward. If N is large or VRAM is tight, this is run in chunks
  of ``--joint-chunk`` (default 8192).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch


def _log(msg: str) -> None:
    """Print a line to stdout and flush so parent processes see progress immediately."""
    print(msg, flush=True)


def _embed_singleton(
    base: "OmnicellBase",
    X_shared: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """``(B, 1, G)`` chunks with ``return_cell_embeddings=False`` → ``lat`` is per-cell.

    Mirrors ``scFM_eval/debug_encoder.py::encode_singleton_sets`` and
    ``debug_omnicell_direct.py``. For ``S=1`` the set-level ``lat`` equals the
    per-cell readout, so we avoid the extra return head.
    """
    device = base.device
    encoder = base.encoder
    n_cells = X_shared.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        x_tensor = torch.from_numpy(X_shared[start:end]).to(device).unsqueeze(1)
        with torch.no_grad():
            lat = encoder(x_tensor, return_cell_embeddings=False)
        chunks.append(lat.float().cpu().numpy())
        bi = len(chunks)
        if bi == 1 or bi == n_batches or bi % max(1, n_batches // 10) == 0:
            _log(
                f"Omnicell embed (singleton): batch {bi}/{n_batches} "
                f"cells {end}/{n_cells} ({time.perf_counter() - t0:.1f}s elapsed)"
            )
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _embed_joint(
    base: "OmnicellBase",
    X_shared: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """``(1, N, G)`` in one or more chunks with ``return_cell_embeddings=True``.

    SetNorm sees ``N`` cells per chunk in a single forward (bigger → closer to the
    "full dataset as one set" regime). Output order preserves the input order.
    Mirrors ``debug_encoder.py::encode_joint_set_per_cell_readout`` when ``N <= chunk_size``.
    """
    device = base.device
    encoder = base.encoder
    n_cells = X_shared.shape[0]
    t0 = time.perf_counter()
    # Probe for latent_dim (needed to preallocate if we ever want to write in-place)
    out: list[np.ndarray] = []
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        x_tensor = (
            torch.from_numpy(X_shared[start:end]).to(device).unsqueeze(0)
        )  # (1, K, G)
        with torch.no_grad():
            res = encoder(x_tensor, return_cell_embeddings=True)
        if not isinstance(res, tuple) or len(res) < 2:
            raise RuntimeError(
                "Joint method requires return_cell_embeddings=True to return (lat, cell_emb, ...)"
            )
        cell_emb = res[1]  # (1, K, L)
        out.append(cell_emb.squeeze(0).float().cpu().numpy())
        bi = len(out)
        if bi == 1 or bi == n_chunks or bi % max(1, n_chunks // 10) == 0:
            _log(
                f"Omnicell embed (joint): chunk {bi}/{n_chunks} "
                f"cells {end}/{n_cells} ({time.perf_counter() - t0:.1f}s elapsed)"
            )
    return np.concatenate(out, axis=0).astype(np.float32)


def _get_cell_embeddings(
    base: "OmnicellBase",
    X: np.ndarray,
    shared_gene_ids: list[int],
    batch_size: int,
    encoding_method: str,
    joint_chunk: int,
) -> np.ndarray:
    """Dispatch to the chosen encoding method; slice shared-gene columns first."""
    X_shared = X[:, shared_gene_ids].astype(np.float32)
    _log(
        f"Omnicell embed: method={encoding_method}  X_shared shape={X_shared.shape}"
    )
    if encoding_method == "singleton":
        return _embed_singleton(base, X_shared, batch_size)
    if encoding_method == "joint":
        return _embed_joint(base, X_shared, chunk_size=joint_chunk)
    raise ValueError(
        f"Unknown encoding_method: {encoding_method!r} (expected one of: singleton, joint)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Omnicell embedding (subprocess entrypoint)")
    parser.add_argument("--input", required=True, help="Input .h5ad path")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--checkpoint_path", required=True, help="Omnicell checkpoint .pt")
    parser.add_argument("--base_dir", required=True, help="cell_types package dir (cell-types/cell_types)")
    parser.add_argument("--config", default="obs", help="Cell-types config name")
    parser.add_argument("--gene_types", default="feature_name", choices=["feature_name", "feature_id"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--encoding-method",
        dest="encoding_method",
        choices=["singleton", "joint"],
        default="singleton",
        help=(
            "singleton: (B,1,G) per-cell (default); joint: (1,N,G) joint set-level — "
            "SetNorm sees all N cells in one forward."
        ),
    )
    parser.add_argument(
        "--joint-chunk",
        dest="joint_chunk",
        type=int,
        default=8192,
        help="For --encoding-method joint, max cells per (1,K,G) forward (default 8192).",
    )
    parser.add_argument(
        "--load-avg",
        dest="load_avg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load averaged weights (avg_encoder_state_dict) when True; "
            "False loads raw encoder_state_dict. Default True, matching "
            "generate_distribution_embeddings.py / cell-types load_encoder."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        print(f"error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    base_path = Path(args.base_dir).resolve()
    if (base_path / "cell_types").is_dir():
        repo_root = base_path
        package_dir = base_path / "cell_types"
    elif base_path.name == "cell_types":
        repo_root = base_path.parent
        package_dir = base_path
    else:
        repo_root = base_path.parent
        package_dir = base_path

    # Match generate_distribution_embeddings.py: generative/ must be on sys.path so Hydra
    # can resolve _target_ keys like generator.count.CountFlow (top-level "generator" package).
    generative_dir = package_dir / "generative"
    path_add = [str(repo_root), str(package_dir)]
    if (package_dir / "bulk_prediction").is_dir():
        path_add.append(str(package_dir / "bulk_prediction"))
    if generative_dir.is_dir():
        path_add.append(str(generative_dir))
    for p in path_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    _log("[omnicell_embed] Importing OmnicellBase (may take a bit)...")
    t_import = time.perf_counter()
    from cell_types.tasks.core.base import OmnicellBase

    _log(f"[omnicell_embed] Import done in {time.perf_counter() - t_import:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"[omnicell_embed] Device: {device}; loading checkpoint (slow step)...")
    t_load = time.perf_counter()
    omnicell_base = OmnicellBase(
        config=args.config,
        checkpoint=str(checkpoint_path),
        pert_config=None,
        pert_checkpoint=None,
        device=device,
        load_avg=args.load_avg,
        base_dir=str(package_dir),
    )
    _log(f"[omnicell_embed] Model ready in {time.perf_counter() - t_load:.1f}s")

    _log(f"[omnicell_embed] Reading {args.input} ...")
    t_read = time.perf_counter()
    adata = ad.read_h5ad(args.input)
    if hasattr(adata, "filename") and adata.isbacked:
        adata = adata.to_memory()
    adata = adata.copy()

    # used_genes: from generator (set at training load) or from checkpoint for inference
    used_genes = getattr(omnicell_base.generator.model, "used_genes", None)
    if used_genes is None:
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            used_genes = ckpt.get("used_genes")
        except Exception:
            pass
    if used_genes is None:
        print("Omnicell: used_genes not on generator and not in checkpoint; map_genes will not filter by used_genes.", file=sys.stderr)

    _log(
        f"[omnicell_embed] AnnData in memory in {time.perf_counter() - t_read:.1f}s "
        f"shape {adata.n_obs} x {adata.n_vars}; mapping genes..."
    )
    t_map = time.perf_counter()
    adata_mapped = omnicell_base.map_genes(adata, args.gene_types, used_genes=used_genes)
    _log(f"[omnicell_embed] Gene map done in {time.perf_counter() - t_map:.1f}s")
    X = adata_mapped.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    _log(
        f"[omnicell_embed] Running encoder on {X.shape[0]} cells "
        f"(method={args.encoding_method}, batch_size={args.batch_size}, "
        f"joint_chunk={args.joint_chunk})..."
    )
    embeddings = _get_cell_embeddings(
        omnicell_base,
        X,
        omnicell_base.shared_gene_ids,
        args.batch_size,
        encoding_method=args.encoding_method,
        joint_chunk=args.joint_chunk,
    )
    _log(f"[omnicell_embed] Embeddings shape {embeddings.shape}; writing {args.output}")
    np.save(args.output, embeddings.astype(np.float32))
    _log("[omnicell_embed] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
