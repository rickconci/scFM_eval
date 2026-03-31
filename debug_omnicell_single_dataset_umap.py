#!/usr/bin/env python3
"""Run Omnicell on every evaluation dataset and plot UMAPs colored by bio / batch.

For each dataset the script:
  1. Loads the ``.h5ad`` and logs detailed gene-format diagnostics (Ensembl vs.
     gene-name, ``feature_name`` / ``feature_id`` presence, mapping rate against
     the global gene list used by the checkpoint).
  2. Logs normalization diagnostics (raw counts vs. normalized, value ranges,
     sparsity, integer-ness).
  3. Runs the Omnicell encoder (via ``OmnicellExtractor``) with the same gene
     resolution path the production pipeline uses.
  4. Produces pre-Harmony UMAPs colored by cell type and batch.
  5. (Optionally) produces post-Harmony UMAPs.

Outputs are written to ``<out-dir>/<dataset_name>/``.

Example::

    cd /orcd/data/omarabu/001/rconci/scFM_eval
    export $(grep -v '^#' .env | xargs)
    python debug_omnicell_single_dataset_umap.py --max-cells 5000

Pass ``--datasets dkd hypomap`` to run only a subset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

_REPO_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry: name → (h5ad filename, batch_col, bio_col)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_REGISTRY: dict[str, dict[str, str]] = {
    "dkd": {"file": "dkd.h5ad", "batch_col": "assay", "bio_col": "cell_type"},
    "hypomap": {"file": "hypomap.h5ad", "batch_col": "assay", "bio_col": "cell_type"},
    "lung_atlas": {"file": "lung_atlas.h5ad", "batch_col": "batch", "bio_col": "cell_type"},
    "gtex_v9": {"file": "gtex_v9.h5ad", "batch_col": "donor_id", "bio_col": "cell_type"},
    "lymph_node_atlas": {
        "file": "lymph_node_atlas.h5ad",
        "batch_col": "donor_id",
        "bio_col": "cell_type",
    },
}

DATA_ROOT = Path("/orcd/data/omarabu/001/Omnicell_datasets/bio_batch_eval_data")
GLOBAL_GENE_LIST = Path(
    "/orcd/data/omarabu/001/Omnicell_datasets/protocol_embeddings/genes/global_gene_mapping.parquet"
)


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────
def _try_load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
    except ImportError:
        logger.info("python-dotenv not installed; export vars manually.")


# ──────────────────────────────────────────────────────────────────────────────
# Gene-format diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def _looks_like_ensembl(names: pd.Index | np.ndarray, sample: int = 200) -> float:
    """Return fraction of a sample that look like Ensembl gene IDs (ENSG…)."""
    subset = np.asarray(names)[:sample]
    return float(np.mean([str(s).startswith("ENSG") for s in subset]))


def diagnose_gene_format(adata: ad.AnnData, dataset_name: str) -> None:
    """Log detailed gene-format diagnostics for *adata*."""
    sep = "=" * 72
    logger.info("%s", sep)
    logger.info("GENE-FORMAT DIAGNOSTICS: %s", dataset_name)
    logger.info("%s", sep)

    var = adata.var
    logger.info("var.index.name  = %r", var.index.name)
    logger.info("var.index.dtype = %s", var.index.dtype)
    logger.info("var.columns     = %s", list(var.columns))
    logger.info("n_vars          = %d", adata.n_vars)

    idx_sample = var.index[:10].tolist()
    logger.info("var.index[:10]  = %s", idx_sample)

    ensembl_frac_idx = _looks_like_ensembl(var.index)
    logger.info(
        "var.index Ensembl fraction (first 200): %.1f%%  →  %s",
        ensembl_frac_idx * 100,
        "ENSEMBL IDs" if ensembl_frac_idx > 0.8 else "gene NAMES" if ensembl_frac_idx < 0.2 else "MIXED",
    )

    for col in ("feature_name", "feature_id", "gene_symbol", "gene_id"):
        if col in var.columns:
            sample = var[col].astype(str).head(5).tolist()
            ensembl_frac = _looks_like_ensembl(var[col].astype(str).values)
            logger.info(
                "var['%s'] present  →  sample %s  (Ensembl %.0f%%)",
                col, sample, ensembl_frac * 100,
            )
        else:
            logger.info("var['%s'] MISSING", col)

    # What the extractor will use for gene_types="feature_name"
    if "gene_symbol" in var.columns:
        resolved = var["gene_symbol"].astype(str).values
        source = "gene_symbol"
    elif "feature_name" in var.columns:
        resolved = var["feature_name"].astype(str).values
        source = "feature_name"
    else:
        resolved = np.asarray(var.index, dtype=str)
        source = "var.index (fallback)"

    logger.info(
        "Extractor will resolve var_names from: %s  →  sample %s",
        source, list(resolved[:5]),
    )

    # Map against the global gene list
    if GLOBAL_GENE_LIST.exists():
        glist = pd.read_parquet(GLOBAL_GENE_LIST)
        global_names = set(glist["feature_name"].values)
        global_ids = set(glist["feature_id"].values)

        matched_names = sum(1 for g in resolved if g in global_names)
        matched_ids = sum(1 for g in var.index if g in global_ids)
        matched_resolved_as_ids = sum(1 for g in resolved if g in global_ids)

        logger.info(
            "Global gene list has %d genes (feature_name unique=%d, feature_id unique=%d)",
            len(glist), len(global_names), len(global_ids),
        )
        logger.info(
            "Resolved names → global feature_name match: %d / %d (%.1f%%)",
            matched_names, len(resolved), 100.0 * matched_names / max(len(resolved), 1),
        )
        logger.info(
            "var.index → global feature_id match: %d / %d (%.1f%%)",
            matched_ids, adata.n_vars, 100.0 * matched_ids / max(adata.n_vars, 1),
        )
        if matched_resolved_as_ids > 0:
            logger.warning(
                "Resolved names that look like feature_id (Ensembl) in global list: %d  "
                "— these will NOT match feature_name and will be dropped!",
                matched_resolved_as_ids,
            )

        unmatched_sample = [g for g in resolved[:100] if g not in global_names][:10]
        if unmatched_sample:
            logger.info("Sample unmatched resolved names: %s", unmatched_sample)
    else:
        logger.warning("Global gene list not found at %s; skipping mapping check.", GLOBAL_GENE_LIST)

    logger.info("%s", sep)


# ──────────────────────────────────────────────────────────────────────────────
# Normalization diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def diagnose_normalization(adata: ad.AnnData, dataset_name: str) -> None:
    """Log normalization / count-type diagnostics."""
    sep = "=" * 72
    logger.info("%s", sep)
    logger.info("NORMALIZATION DIAGNOSTICS: %s", dataset_name)
    logger.info("%s", sep)

    X = adata.X
    is_sparse = issparse(X)
    logger.info("X type: %s  sparse=%s", type(X).__name__, is_sparse)

    if is_sparse:
        data_vals = X.data
    else:
        data_vals = np.asarray(X).ravel()

    if len(data_vals) == 0:
        logger.warning("X is completely empty!")
        return

    sample = data_vals[:10000]
    logger.info("X.dtype = %s", X.dtype)
    logger.info("Value range: [%.4f, %.4f]", np.min(sample), np.max(sample))
    logger.info("Mean (sample): %.4f   Median: %.4f", np.mean(sample), np.median(sample))

    # Check integer-ness (raw counts should be integers)
    if np.issubdtype(sample.dtype, np.floating):
        int_frac = np.mean(np.equal(np.mod(sample, 1), 0))
        logger.info(
            "Integer fraction (sample): %.1f%%  →  %s",
            int_frac * 100,
            "likely RAW COUNTS" if int_frac > 0.95 else "likely NORMALIZED/log-transformed",
        )
    else:
        logger.info("dtype is integer → RAW COUNTS")

    # Sparsity
    if is_sparse:
        nnz = X.nnz
        total = X.shape[0] * X.shape[1]
        logger.info("Sparsity: %.2f%% zeros  (%d / %d nonzero)", 100 * (1 - nnz / total), nnz, total)
    else:
        nz = np.count_nonzero(data_vals)
        logger.info("Nonzero: %d / %d", nz, len(data_vals))

    # Negative values (shouldn't appear in raw counts)
    neg_count = int(np.sum(sample < 0))
    if neg_count > 0:
        logger.warning("Found %d NEGATIVE values in sample — unexpected for raw counts!", neg_count)

    # Per-cell library size
    if is_sparse:
        lib_sizes = np.asarray(X.sum(axis=1)).ravel()
    else:
        lib_sizes = np.sum(X, axis=1)
    logger.info(
        "Per-cell library size: min=%.1f, median=%.1f, max=%.1f, std=%.1f",
        np.min(lib_sizes), np.median(lib_sizes), np.max(lib_sizes), np.std(lib_sizes),
    )
    # Constant library size is a sign of size-factor normalization
    cv = np.std(lib_sizes) / max(np.mean(lib_sizes), 1e-12)
    if cv < 0.01:
        logger.warning(
            "Library sizes have CV=%.4f → nearly constant → likely SIZE-FACTOR NORMALIZED (not raw)!", cv
        )
    else:
        logger.info("Library size CV=%.4f → consistent with raw counts", cv)

    logger.info("%s", sep)


# ──────────────────────────────────────────────────────────────────────────────
# Subsampling
# ──────────────────────────────────────────────────────────────────────────────
def _subsample_adata(adata: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    if adata.n_obs <= max_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
    idx.sort()
    return adata[idx].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _categorical_colors(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    uniq = sorted(set(str(x) for x in labels if str(x) != "nan"))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color_list = [cmap(i % cmap.N) for i in range(len(uniq))]
    lut = {u: color_list[i] for i, u in enumerate(uniq)}
    rgba = np.array(
        [lut.get(str(l), (0.5, 0.5, 0.5, 1.0)) for l in labels],
        dtype=float,
    )
    return rgba, uniq


def _plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    rgba, uniq = _categorical_colors(labels)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=rgba, s=2, alpha=0.85, linewidths=0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=u,
            markerfacecolor=_categorical_colors(np.array([u]))[0][0][:3],
            markersize=6,
        )
        for u in uniq[:20]
    ]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _umap_from_matrix(
    x: np.ndarray,
    cell_type_labels: np.ndarray,
    batch: np.ndarray,
    prefix: str,
    out_dir: Path,
) -> None:
    n_comp = min(50, x.shape[1], x.shape[0] - 1)
    if n_comp < 2:
        raise ValueError(f"Not enough dimensions/samples for PCA (n_comp={n_comp}).")
    tmp = ad.AnnData(X=x.astype(np.float32))
    tmp.obs["cell_type_plot"] = cell_type_labels.astype(str)
    tmp.obs["batch_plot"] = batch.astype(str)
    sc.tl.pca(tmp, n_comps=n_comp, svd_solver="arpack")
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=n_comp, use_rep="X_pca")
    sc.tl.umap(tmp, min_dist=0.3)
    u = tmp.obsm["X_umap"]
    _plot_umap(u, tmp.obs["cell_type_plot"].to_numpy(), f"{prefix} UMAP (cell type)", out_dir / f"{prefix}_cell_type.png")
    _plot_umap(u, tmp.obs["batch_plot"].to_numpy(), f"{prefix} UMAP (batch)", out_dir / f"{prefix}_batch.png")


def _harmony_then_umap(
    x: np.ndarray,
    batch_labels: np.ndarray,
    cell_type_labels: np.ndarray,
    batch: np.ndarray,
    prefix: str,
    out_dir: Path,
) -> bool:
    try:
        import scanpy.external.pp as scext
    except ImportError:
        logger.warning("scanpy.external not available; skipping Harmony.")
        return False
    if not hasattr(scext, "harmony_integrate"):
        logger.warning("harmony_integrate not found; skipping post-Harmony UMAP.")
        return False
    n_comp = min(50, x.shape[1], x.shape[0] - 1)
    if n_comp < 2:
        return False
    tmp = ad.AnnData(X=x.astype(np.float32))
    tmp.obs["cell_type_plot"] = cell_type_labels.astype(str)
    tmp.obs["batch_plot"] = batch.astype(str)
    tmp.obs["batch_harmony"] = batch_labels.astype(str)
    sc.tl.pca(tmp, n_comps=n_comp, svd_solver="arpack")
    try:
        scext.harmony_integrate(tmp, key="batch_harmony", basis="X_pca", adjusted_key="X_pca_harmony")
    except Exception as exc:
        logger.warning("Harmony failed (%s); skipping post-Harmony plots.", exc)
        return False
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=n_comp, use_rep="X_pca_harmony")
    sc.tl.umap(tmp, min_dist=0.3)
    u = tmp.obsm["X_umap"]
    _plot_umap(u, tmp.obs["cell_type_plot"].to_numpy(), f"{prefix} UMAP (cell type)", out_dir / f"{prefix}_cell_type.png")
    _plot_umap(u, tmp.obs["batch_plot"].to_numpy(), f"{prefix} UMAP (batch)", out_dir / f"{prefix}_batch.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_dataset(
    dataset_name: str,
    cfg: dict[str, str],
    *,
    ckpt: str,
    base_dir: str,
    gene_types: str,
    batch_size: int,
    max_cells: int,
    seed: int,
    skip_harmony: bool,
    out_root: Path,
) -> None:
    """Full diagnostic + embedding + UMAP pipeline for one dataset."""
    h5ad_path = DATA_ROOT / cfg["file"]
    bio_col = cfg["bio_col"]
    batch_col = cfg["batch_col"]

    banner = f"  DATASET: {dataset_name}  "
    logger.info("\n\n%s\n%s\n%s", "#" * 80, banner.center(80, "#"), "#" * 80)
    logger.info("h5ad: %s", h5ad_path)
    logger.info("bio_col=%s  batch_col=%s", bio_col, batch_col)

    if not h5ad_path.is_file():
        logger.error("File not found: %s — skipping %s", h5ad_path, dataset_name)
        return

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Reading %s ...", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    if hasattr(adata, "isbacked") and adata.isbacked:
        adata = adata.to_memory()
    logger.info("Loaded: %d cells × %d genes", adata.n_obs, adata.n_vars)

    if bio_col not in adata.obs.columns:
        logger.error("bio_col %r not in obs.columns %s — skipping", bio_col, list(adata.obs.columns))
        return
    if batch_col not in adata.obs.columns:
        logger.error("batch_col %r not in obs.columns %s — skipping", batch_col, list(adata.obs.columns))
        return

    logger.info(
        "bio (%s): %d unique values,  batch (%s): %d unique values",
        bio_col, adata.obs[bio_col].nunique(),
        batch_col, adata.obs[batch_col].nunique(),
    )

    # ── Extract raw counts (mirrors data_loader.py load_raw=True) ─────────
    # CELLxGENE h5ads store log1p-normalized in X and raw counts in .raw.X.
    # The production pipeline (H5ADLoader with load_raw=True) calls
    # adata = adata.raw.to_adata() — we replicate that here.
    if adata.raw is not None:
        logger.info(
            "adata.raw exists (shape %d × %d); using adata.raw.to_adata() "
            "to get raw counts (matches data_loader load_raw=True).",
            adata.raw.n_vars, adata.raw.n_vars,
        )
        # Diagnostics on the ORIGINAL X before swapping (so we can see it was log1p)
        diagnose_normalization(adata, f"{dataset_name} [ORIGINAL adata.X — before raw swap]")
        adata = adata.raw.to_adata()
        logger.info("After raw swap: %d cells × %d genes", adata.n_obs, adata.n_vars)
    else:
        logger.warning("adata.raw is None — no raw layer available.")
        # Detect log1p and reverse if needed
        X_sample = adata.X[:min(500, adata.n_obs)]
        if issparse(X_sample):
            X_sample = X_sample.toarray()
        X_sample = np.asarray(X_sample, dtype=np.float32)
        nz = X_sample[X_sample != 0]
        if len(nz) > 0:
            int_frac = float(np.mean(np.equal(np.mod(nz, 1), 0)))
            min_nz = float(np.min(nz))
            if int_frac < 0.1 and abs(min_nz - np.log1p(1)) < 0.01:
                logger.warning(
                    "Data looks log1p-transformed (min_nz=%.4f ≈ ln(2), int_frac=%.1f%%). "
                    "Applying expm1 to recover raw counts.",
                    min_nz, int_frac * 100,
                )
                if issparse(adata.X):
                    adata.X = adata.X.copy()
                    adata.X.data = np.expm1(adata.X.data)
                else:
                    adata.X = np.expm1(adata.X)

    # ── Diagnostics (after raw swap / expm1) ──────────────────────────────
    diagnose_gene_format(adata, dataset_name)
    diagnose_normalization(adata, f"{dataset_name} [fed to encoder]")

    # ── Subsample ─────────────────────────────────────────────────────────
    if max_cells and max_cells > 0 and adata.n_obs > max_cells:
        adata = _subsample_adata(adata, max_cells, seed)
        logger.info("Subsampled to %d cells", adata.n_obs)

    cell_type_labels = adata.obs[bio_col].to_numpy()
    batch_labels = adata.obs[batch_col].to_numpy()

    # ── Omnicell embedding ────────────────────────────────────────────────
    sys.path.insert(0, str(_REPO_ROOT / "extractors"))
    from omnicell.extract import OmnicellExtractor

    extractor = OmnicellExtractor(
        checkpoint_path=ckpt,
        base_dir=base_dir,
        config="obs",
        batch_size=batch_size,
        device="auto",
        gene_types=gene_types,
        load_avg=True,
    )
    logger.info("Running Omnicell extract_embeddings (%d cells) ...", adata.n_obs)
    emb = extractor.extract_embeddings(adata)
    logger.info("Embeddings shape: %s  dtype: %s", emb.shape, emb.dtype)

    # Quick sanity on embedding values
    logger.info(
        "Embedding stats: min=%.4f  max=%.4f  mean=%.4f  std=%.4f  any_nan=%s  any_inf=%s",
        np.min(emb), np.max(emb), np.mean(emb), np.std(emb),
        bool(np.any(np.isnan(emb))), bool(np.any(np.isinf(emb))),
    )
    zero_rows = int(np.sum(np.all(emb == 0, axis=1)))
    if zero_rows > 0:
        logger.warning(
            "%d / %d cells produced ALL-ZERO embeddings (%.1f%%)!",
            zero_rows, emb.shape[0], 100.0 * zero_rows / emb.shape[0],
        )
    const_rows = int(np.sum(np.std(emb, axis=1) < 1e-8))
    if const_rows > 0:
        logger.warning(
            "%d / %d cells have near-constant (std<1e-8) embeddings",
            const_rows, emb.shape[0],
        )

    # ── Save + UMAP ──────────────────────────────────────────────────────
    ds_out = out_root / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)
    np.save(ds_out / "omnicell_embeddings.npy", emb)
    logger.info("Saved embeddings to %s", ds_out / "omnicell_embeddings.npy")

    logger.info("Computing pre-Harmony UMAP ...")
    _umap_from_matrix(emb, cell_type_labels, batch_labels, "pre_harmony", ds_out)

    if not skip_harmony:
        logger.info("Computing post-Harmony UMAP ...")
        ok = _harmony_then_umap(emb, batch_labels, cell_type_labels, batch_labels, "post_harmony", ds_out)
        if not ok:
            logger.info("Harmony unavailable; only pre-Harmony plots written.")

    logger.info("Done with %s. Outputs: %s", dataset_name, ds_out)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Subset of datasets to run (default: all). Choices: {list(DATASET_REGISTRY)}",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("./debug_omnicell_umap"),
        help="Root output directory.",
    )
    p.add_argument("--max-cells", type=int, default=5_000, help="Per-dataset subsample size (0=all).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    p.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Omnicell .pt checkpoint (default: OMNICELL_CHECKPOINT_PATH from env).",
    )
    p.add_argument("--base-dir", type=str, default=None, help="cell_types package dir.")
    p.add_argument(
        "--gene-types", type=str, choices=("feature_name", "feature_id"),
        default="feature_name",
    )
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--skip-harmony", action="store_true")
    return p.parse_args()


def main() -> int:
    _try_load_dotenv()
    args = _parse_args()

    ckpt = args.checkpoint_path or os.environ.get("OMNICELL_CHECKPOINT_PATH")
    base_dir = args.base_dir or os.environ.get("OMNICELL_BASE_DIR")
    if not ckpt or not Path(ckpt).is_file():
        logger.error("Missing checkpoint (--checkpoint-path or OMNICELL_CHECKPOINT_PATH). Got: %s", ckpt)
        return 2
    if not base_dir or not Path(base_dir).is_dir():
        logger.error("Missing base_dir (--base-dir or OMNICELL_BASE_DIR). Got: %s", base_dir)
        return 2

    logger.info("Checkpoint : %s", ckpt)
    logger.info("Base dir   : %s", base_dir)
    logger.info("Gene types : %s", args.gene_types)
    logger.info("Max cells  : %s", args.max_cells)

    datasets_to_run = args.datasets or list(DATASET_REGISTRY)
    for name in datasets_to_run:
        if name not in DATASET_REGISTRY:
            logger.error("Unknown dataset %r. Choices: %s", name, list(DATASET_REGISTRY))
            return 2

    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets_to_run:
        try:
            process_dataset(
                ds_name,
                DATASET_REGISTRY[ds_name],
                ckpt=ckpt,
                base_dir=base_dir,
                gene_types=args.gene_types,
                batch_size=args.batch_size,
                max_cells=args.max_cells,
                seed=args.seed,
                skip_harmony=args.skip_harmony,
                out_root=out_root,
            )
        except Exception:
            logger.exception("FAILED on dataset %s", ds_name)

    logger.info("All done. Results under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
