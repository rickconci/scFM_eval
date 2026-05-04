#!/usr/bin/env python3
"""Omnicell encoder debug: load encoder, map h5ad, multiple embedding methods, UMAPs + metrics.

1. **Load encoder** — ``generative.utils.loading.load_encoder`` (``obs`` + gene_manager).
2. **Load data +** ``map_genes`` — dense **X_shared** and obs labels.
3. **Method A — singleton sets:** chunked **(B,1,G)** with ``return_cell_embeddings=False``;
   for **S=1** the set-level ``lat`` is the per-cell vector.
4. **Method B — joint set:** **(1,N,G)** with ``return_cell_embeddings=True``; SetNorm
   sees all N cells together.
5. **Method C — grouped-by-label sets:** per cell_type, **(1, n_ct, G)** with
   ``return_cell_embeddings=True``. This matches how Omnicell is trained (sets of cells
   from the same biological group) — the SetNorm statistics then characterize a single
   cell type rather than a mixture.

Also computes batch + bio metrics (ASW_label, ASW_batch, iLISI, cLISI) — the same
definitions used in ``scFM_eval/evaluation/{eval,batch_effects}.py`` — and saves them
to ``metrics.csv`` for each method plus a **PCA baseline** on the same ``X_shared``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

# Paths: adjust if your tree differs
_REPO = Path(__file__).resolve().parent
CELL_TYPES_ROOT = Path("/orcd/data/omarabu/001/rconci/cell-types")
PACKAGE_DIR = CELL_TYPES_ROOT / "cell_types"
DEFAULT_DKD = Path(
    "/orcd/scratch/bcs/002/njwfish/Omnicell_datasets/bio_batch_eval_data/dkd.h5ad"
)
DEFAULT_CHECKPOINT = (
    "/orcd/scratch/bcs/002/njwfish/cell-types/cell_types/outputs/obs/"
    "517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_5.pt"
)
DEFAULT_MAX_CELLS = 2000
DEFAULT_OUT_BASE = _REPO / "debug_encoder_runs"
BATCH_SIZE_ENCODE = 4096
# Rows used only for (A) vs (B) cosine/RMSE print
LAYOUT_COMPARE_CELLS = 64
BIO_OBS_KEY = "cell_type"
BATCH_OBS_KEY = "assay"


@dataclass(frozen=True)
class DkdEncoderData:
    """Mapped DKD: shared-gene matrix + obs, for encoder + diagnostics."""

    X_shared: np.ndarray
    cell_types: np.ndarray
    batches: np.ndarray
    ad_m: ad.AnnData
    shared_gene_ids: list[int]
    total_gene_ids: np.ndarray
    h5ad_stem: str
    res_source: str
    map_mode: str
    n_ensg: int
    gene_ids_input_only: np.ndarray


def _ensure_path() -> None:
    for p in (str(CELL_TYPES_ROOT), str(PACKAGE_DIR), str(PACKAGE_DIR / "generative")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_used_genes_from_checkpoint(path: Path) -> Optional[set[Any]]:
    if not path.is_file():
        return None
    ck: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)
    ug = ck.get("used_genes")
    if ug is None:
        return None
    return set(ug)


def _resolve_var_names(adata: ad.AnnData, map_mode: str) -> tuple[ad.AnnData, str]:
    if map_mode == "feature_name":
        if "gene_symbol" in adata.var.columns:
            source = "gene_symbol"
            adata.var.index = adata.var["gene_symbol"].astype(str).values
        elif "feature_name" in adata.var.columns:
            source = "feature_name"
            adata.var.index = adata.var["feature_name"].astype(str).values
        else:
            source = "var.index (no feature_name or gene_symbol)"
    elif map_mode == "feature_id":
        if "gene_id" in adata.var.columns:
            source = "gene_id"
            adata.var.index = adata.var["gene_id"].astype(str).values
        elif "feature_id" in adata.var.columns:
            source = "feature_id"
            adata.var.index = adata.var["feature_id"].astype(str).values
        else:
            source = "var.index (no feature_id or gene_id)"
    else:
        raise ValueError("map_mode must be 'feature_name' or 'feature_id'")
    return adata, source


def _load_dkd_adata(h5ad_path: Path, max_cells: int, use_raw: bool) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path)
    if getattr(adata, "isbacked", False) and adata.isbacked:
        adata = adata.to_memory()
    if use_raw and adata.raw is not None:
        adata = adata.raw.to_adata()
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        cells = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        adata = adata[cells].copy()
    return adata


def _obs_str(adata: ad.AnnData, key: str) -> np.ndarray:
    if key not in adata.obs.columns:
        raise KeyError(f"obs['{key}'] missing; columns: {list(adata.obs.columns)[:30]}")
    return adata.obs[key].astype(str).to_numpy()


def load_dkd_encoder_data(
    h5ad_path: Path,
    max_cells: int,
    use_raw: bool,
    map_mode: str,
    gene_manager: dict[str, Any],
    used_genes: Optional[set[Any]],
) -> DkdEncoderData:
    """Load h5ad, run ``map_genes``, return shared expression + obs. Single map for whole script."""
    from constants import GLOBAL_GENE_LIST_PATH
    from tasks.data.genes import map_genes, map_genes_to_global_list

    adata = _load_dkd_adata(h5ad_path, max_cells, use_raw)
    adata, res_source = _resolve_var_names(adata, map_mode)
    adata.var_names_make_unique()
    n_ensg = sum(1 for g in adata.var_names if str(g).upper().startswith("ENSG"))

    gene_ids_raw = map_genes_to_global_list(
        adata.var_names.to_numpy(), map_mode, str(GLOBAL_GENE_LIST_PATH)
    )
    m_ok = gene_ids_raw != -1
    gene_ids_step = gene_ids_raw[m_ok]
    if used_genes is not None:
        u_arr = np.array(list(used_genes))
        in_used = np.isin(gene_ids_step, u_arr)
        gene_ids_input_only = gene_ids_step[in_used]
    else:
        gene_ids_input_only = gene_ids_step

    mapped: dict[str, Any] = map_genes(adata, map_mode, gene_manager, used_genes=used_genes)
    ad_m: ad.AnnData = mapped["adata"]
    s_ids: list[int] = list(mapped["shared_gene_ids"])
    X = ad_m.X
    if issparse(X):
        X = X.toarray()
    Xs = np.asarray(X[:, s_ids], dtype=np.float32)
    return DkdEncoderData(
        X_shared=Xs,
        cell_types=_obs_str(ad_m, BIO_OBS_KEY),
        batches=_obs_str(ad_m, BATCH_OBS_KEY),
        ad_m=ad_m,
        shared_gene_ids=s_ids,
        total_gene_ids=np.asarray(mapped["total_gene_ids"]),
        h5ad_stem=h5ad_path.stem,
        res_source=res_source,
        map_mode=map_mode,
        n_ensg=n_ensg,
        gene_ids_input_only=np.asarray(gene_ids_input_only, dtype=np.int64),
    )


def print_dkd_map_diagnostics(
    data: DkdEncoderData,
    gene_manager: dict[str, Any],
    checkpoint: Path,
) -> None:
    """Map_genes + overlap + sparsity (stdout) using the mapped object from ``load_dkd_encoder_data``."""
    ad_m, shared_gene_ids, total_gene_ids = data.ad_m, data.shared_gene_ids, data.total_gene_ids
    shared_manager: Set[int] = set(gene_manager["shared_genes"])
    h5ad_stem, res_source, map_mode = data.h5ad_stem, data.res_source, data.map_mode
    n_ensg, gene_ids_input_only = data.n_ensg, data.gene_ids_input_only
    n_sym_like = int(ad_m.n_vars) - n_ensg
    n_input_cols = int(gene_ids_input_only.size)
    n_input_in_shared = (
        int(np.isin(gene_ids_input_only, list(shared_manager)).sum()) if n_input_cols else 0
    )
    uniq_m = np.unique(gene_ids_input_only)
    n_uniq, n_uniq_in_shared = int(uniq_m.size), int(
        np.isin(uniq_m, list(shared_manager)).sum()
    )

    print(f"\n--- DKD h5ad ({h5ad_stem}) for map_genes ---")
    print(f"  cells: {ad_m.n_obs}  genes (var, pre-map): n/a in cache  (see load)")
    print(f"  var name resolution: {res_source!r}  (map_gene mode={map_mode!r})")
    print(f"  heuristics: ~{n_ensg} Ensembl-like names, ~{n_sym_like} not (from mapped n_vars)")

    used = _load_used_genes_from_checkpoint(checkpoint)
    if used is not None:
        print(f"  used_genes from checkpoint: {len(used)} entries")

    in_shared = 0
    oob = 0
    for li in shared_gene_ids:
        gid = int(total_gene_ids[li])
        if gid in shared_manager:
            in_shared += 1
        else:
            oob += 1
    in_data: Set[int] = {int(x) for x in total_gene_ids}
    covered = sum(1 for g in shared_manager if g in in_data)

    print("  map_genes output:")
    print(f"    n_vars in mapped adata: {ad_m.n_vars}")
    print(f"    len(shared_gene_ids) = {len(shared_gene_ids)}  (encoder slice width)")
    if len(shared_gene_ids) == len(gene_manager["shared_genes"]):
        print("    matches len(gene_manager['shared_genes'])  ✓")
    print(
        f"    local shared columns in gene_manager[shared]: {in_shared} / {len(shared_gene_ids)}"
    )
    if oob:
        print(f"    WARNING: {oob} oob (unexpected)")
    print(f"    training shared with any column here: {covered} / {len(shared_manager)}")
    print("  input × shared overlap (pre-union):")
    print(
        f"    var columns in [shared]: {n_input_in_shared} / {n_input_cols} "
        f"({(100.0 * n_input_in_shared / n_input_cols if n_input_cols else 0):.1f}%)"
    )
    print(f"    unique global ids: {n_uniq}  in shared: {n_uniq_in_shared} / {len(shared_manager)}")

    n_obs = int(ad_m.n_obs)
    col_idx = np.asarray(shared_gene_ids, dtype=np.intp)
    x_sub = ad_m.X[:, col_idx]
    if issparse(x_sub):
        x_csc = x_sub.tocsc()
        nnz = np.diff(x_csc.indptr).astype(np.intp, copy=False)
    else:
        nnz = np.count_nonzero(x_sub, axis=0)
    fr = np.asarray(nnz, dtype=np.float64) / float(n_obs)
    n_gt_10p = int((fr > 0.1).sum())
    print("  sparsity (shared slice):")
    print(
        f"    >10% cells non-zero: {n_gt_10p} / {len(shared_gene_ids)}  "
        f"({(100.0 * n_gt_10p / max(len(shared_gene_ids), 1)):.1f}%)"
    )
    print(
        f"    per-column cell-nnz fraction: min={fr.min():.3f}  p50={float(np.median(fr)):.3f}  max={fr.max():.3f}"
    )
    is_int_dtype = bool(np.issubdtype(ad_m.X.dtype, np.integer))
    if issparse(x_sub):
        g_max, g_min = float(x_sub.max()), float(x_sub.min())
        stored = np.asarray(x_sub.data, dtype=np.float64)
    else:
        x_den = np.asarray(x_sub, dtype=np.float64)
        g_max, g_min = float(np.max(x_den)), float(np.min(x_den))
        stored = x_den.ravel()
    print("  value stats (shared):")
    print(
        f"    dtype: integer={is_int_dtype}  {ad_m.X.dtype!r}  |  min={g_min:.4g}  max={g_max:.4g}"
    )
    if stored.size:
        int_frac = float(np.mean(np.isclose(stored, np.rint(stored), rtol=0, atol=1e-5)))
        all_int = bool(np.allclose(stored, np.rint(stored), atol=1e-5))
        print(
            f"    stored nonzeros: n={stored.size}  all≈int? {all_int!s}  frac_int≈ {int_frac:.4f}"
        )


def _describe_shared_genes(gene_manager: dict[str, Any]) -> int:
    shared = list(gene_manager["shared_genes"])
    total = gene_manager["total_genes"]
    n_s, n_t = len(shared), len(total)
    print("\n--- gene_manager ---")
    print(f"  len(shared_genes) = {n_s}  (encoder in_dim)")
    print(f"  len(total_genes)  = {n_t}")
    if n_s:
        print(f"  first 5 / last 3 shared-gene *global* indices: {shared[:5]!r} … {shared[-3:]!r}")
    return n_s


def _encoder_in_dim(encoder: Any) -> int | None:
    if hasattr(encoder, "input_projection") and hasattr(encoder.input_projection, "in_features"):
        return int(encoder.input_projection.in_features)
    return None


# ── 3) Encoder: two explicit methods ─────────────────────────────────────────


def encode_singleton_sets(
    encoder: torch.nn.Module,
    X_shared: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """(B,1,G) chunks, ``return_cell_embeddings=False`` → ``lat`` (B, L). For S=1, ``lat`` is per-cell."""
    device = next(encoder.parameters()).device
    n = X_shared.shape[0]
    out_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_t = torch.from_numpy(X_shared[start:end].astype(np.float32)).to(device).unsqueeze(1)
        with torch.no_grad():
            lat = encoder(x_t, return_cell_embeddings=False)
        out_chunks.append(lat.float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def encode_joint_set_per_cell_readout(
    encoder: torch.nn.Module,
    X_shared: np.ndarray,
) -> np.ndarray:
    """(1, n, G) once; take second output (1,n,L) → (n, L). SetNorm couples all n cells."""
    device = next(encoder.parameters()).device
    t = torch.from_numpy(X_shared.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        out = encoder(t, return_cell_embeddings=True)
    if not isinstance(out, tuple) or len(out) < 2:
        raise RuntimeError("Encoder must return (lat, cell_emb, …) with return_cell_embeddings=True")
    cell = out[1]
    return cell.squeeze(0).float().cpu().numpy().astype(np.float32)


def encode_grouped_by_label(
    encoder: torch.nn.Module,
    X_shared: np.ndarray,
    labels: np.ndarray,
    max_set_size: int = 4096,
) -> np.ndarray:
    """One forward per **label** group: ``(1, n_ct, G)`` + ``return_cell_embeddings=True``.

    Within-group SetNorm now sees cells of the *same* cell type — closer to how the
    encoder is trained. If a group has more than ``max_set_size`` cells, it is split
    into disjoint chunks of up to ``max_set_size`` (keeps memory in check on large groups).
    Output rows are written back in **original order** so downstream UMAP/scores align.
    """
    device = next(encoder.parameters()).device
    n = X_shared.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"labels ({labels.shape[0]}) != rows ({n})")
    latent_dim: Optional[int] = None
    with torch.no_grad():
        probe = encoder(
            torch.from_numpy(X_shared[:1].astype(np.float32)).to(device).unsqueeze(0),
            return_cell_embeddings=True,
        )
    latent_dim = int(probe[1].shape[-1])
    out = np.empty((n, latent_dim), dtype=np.float32)

    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    uniq, starts = np.unique(sorted_labels, return_index=True)
    ends = np.r_[starts[1:], len(sorted_labels)]
    print(f"  grouped-by-label: {len(uniq)} groups, sizes: "
          f"min={int((ends-starts).min())}  median={int(np.median(ends-starts))}  max={int((ends-starts).max())}")

    for ct, s, e in zip(uniq, starts, ends):
        idx_grp = order[s:e]
        # chunk large groups
        for cstart in range(0, len(idx_grp), max_set_size):
            cidx = idx_grp[cstart : cstart + max_set_size]
            x_t = torch.from_numpy(X_shared[cidx].astype(np.float32)).to(device).unsqueeze(0)
            with torch.no_grad():
                _lat, cell_emb, _ = encoder(x_t, return_cell_embeddings=True)
            out[cidx] = cell_emb.squeeze(0).float().cpu().numpy()
    return out


def print_layout_compare(
    emb_singleton: np.ndarray, emb_joint: np.ndarray, n: int, g_dim: int
) -> None:
    k = int(min(n, emb_singleton.shape[0], emb_joint.shape[0]))
    a, b = emb_singleton[:k], emb_joint[:k]
    d = a - b
    rmse = float(np.sqrt(np.mean(d**2)))
    an = np.linalg.norm(a, axis=1) + 1e-12
    bn = np.linalg.norm(b, axis=1) + 1e-12
    cos = np.sum(a * b, axis=1) / (an * bn)
    print(f"\n--- layout compare: first {k} cells, G={g_dim} ---")
    print("  (A) singleton: (batch,1,G) + return_cell_embeddings=False → lat (chunked) — *recommended*")
    print("  (B) joint set: (1,N,G) + return_cell_embeddings — SetNorm over all N cells in one forward")
    print(f"  per-cell cos(A,B):  mean={cos.mean():.4f}  min={cos.min():.4f}  p50={float(np.median(cos)):.4f}  max={cos.max():.4f}")
    print(f"  L2 row RMSE: {rmse:.6f}")


def write_umap_figures(
    out_dir: Path,
    title_prefix: str,
    emb: np.ndarray,
    cell_types: np.ndarray,
    batches: np.ndarray,
) -> None:
    """UMAPs via ``sc.pl.umap`` — same call pattern as ``viz/visualization.EmbeddingVisualizer``."""
    from matplotlib import pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    # arpack requires n_components < min(n_samples, n_features)
    n_comp = max(2, min(50, emb.shape[1] - 1, emb.shape[0] - 1))
    tmp = ad.AnnData(X=emb.astype(np.float32))
    # Match pipeline: biological = ``label``, batch = ``batch`` (see ``viz/visualization.py``)
    tmp.obs["label"] = cell_types.astype(str)
    tmp.obs["batch"] = batches.astype(str)
    sc.tl.pca(tmp, n_comps=n_comp, svd_solver="arpack")
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=n_comp, use_rep="X_pca")
    sc.tl.umap(tmp, min_dist=0.3)

    for col, out_name in [
        ("batch", "embedding_batch.png"),
        ("label", "embedding_label.png"),
    ]:
        # Same kwargs as ``EmbeddingVisualizer.plot`` PRE UMAP figures
        fig = sc.pl.umap(
            tmp,
            color=col,
            show=False,
            wspace=0.4,
            frameon=False,
            return_fig=True,
        )
        p = out_dir / out_name
        f = fig[0] if isinstance(fig, (list, tuple)) and fig is not None else fig
        if f is None:
            raise RuntimeError("sc.pl.umap(..., return_fig=True) did not return a figure")
        f.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(f)
        print(f"  Wrote {p}  ({title_prefix})")


def _compute_lisi_pure_python(adata: ad.AnnData, obs_key: str, n_neighbors: int = 90) -> np.ndarray:
    """Pure-Python LISI over an existing neighbor graph — see ``scFM_eval/evaluation/batch_effects.py``."""
    import scipy.sparse as sp

    if "connectivities" not in adata.obsp:
        raise ValueError("Need precomputed neighbors (sc.pp.neighbors).")
    conn = adata.obsp["connectivities"]
    if not sp.issparse(conn):
        raise ValueError("Expected sparse connectivities matrix.")
    labels = adata.obs[obs_key].to_numpy()
    _, label_idx = np.unique(labels, return_inverse=True)
    n_lab = int(label_idx.max()) + 1
    n = adata.n_obs
    scores = np.empty(n, dtype=float)
    for i in range(n):
        row = conn.getrow(i)
        if row.nnz == 0:
            scores[i] = 1.0
            continue
        w = row.data
        w = w / w.sum()
        props = np.bincount(label_idx[row.indices], weights=w, minlength=n_lab)
        s = float(np.sum(props**2))
        scores[i] = 1.0 / s if s > 0 else 1.0
    _ = n_neighbors
    return scores


def compute_embedding_scores(
    name: str,
    emb: np.ndarray,
    cell_types: np.ndarray,
    batches: np.ndarray,
    n_neighbors: int = 90,
    subsample: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Batch+bio scores: ``ASW_label``, ``ASW_batch``, ``iLISI`` (batch), ``cLISI`` (cell type).

    Mirrors the metric definitions in ``scFM_eval/evaluation/eval.py`` and
    ``batch_effects.py``. Returns raw LISI medians (higher iLISI = better batch mixing,
    lower cLISI = better cell-type separation); silhouettes are rescaled to [0, 1].
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    if subsample and n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        idx.sort()
        emb, cell_types, batches = emb[idx], cell_types[idx], batches[idx]

    out: dict[str, float] = {}
    # ── ASW (rescaled to [0,1] as in scib.metrics.silhouette) ──
    try:
        sl = float(silhouette_score(emb, cell_types, metric="euclidean"))
        out["ASW_label"] = (sl + 1.0) / 2.0
    except Exception as e:
        print(f"    ASW_label failed: {e}")
    if len(np.unique(batches)) > 1:
        try:
            sb = float(silhouette_score(emb, batches, metric="euclidean"))
            out["ASW_batch"] = 1.0 - (sb + 1.0) / 2.0  # lower batch-silhouette = better
        except Exception as e:
            print(f"    ASW_batch failed: {e}")
    # ── LISI (pure-python over scanpy neighbor graph) ──
    tmp = ad.AnnData(X=emb.astype(np.float32))
    tmp.obs["label"] = cell_types.astype(str)
    tmp.obs["batch"] = batches.astype(str)
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=n_neighbors, metric="euclidean")
    if len(np.unique(batches)) > 1:
        try:
            out["iLISI_med"] = float(np.nanmedian(_compute_lisi_pure_python(tmp, "batch", n_neighbors)))
        except Exception as e:
            print(f"    iLISI failed: {e}")
    try:
        out["cLISI_med"] = float(np.nanmedian(_compute_lisi_pure_python(tmp, "label", n_neighbors)))
    except Exception as e:
        print(f"    cLISI failed: {e}")
    out["method"] = name  # type: ignore[assignment]
    return out


def _pca_baseline(X: np.ndarray, n_comp: int = 50) -> np.ndarray:
    """Plain Scanpy PCA on ``X_shared`` (reference point for the encoder UMAPs/metrics)."""
    tmp = ad.AnnData(X=X.astype(np.float32))
    n_c = int(min(n_comp, X.shape[1], X.shape[0] - 1))
    sc.tl.pca(tmp, n_comps=n_c, svd_solver="arpack")
    return np.asarray(tmp.obsm["X_pca"], dtype=np.float32)


def write_metrics_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one row per method (keys = method, ASW_label, ASW_batch, iLISI_med, cLISI_med)."""
    import csv

    cols = ["method", "ASW_label", "ASW_batch", "iLISI_med", "cLISI_med"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n  Wrote {out_path}")


# ── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", type=str, default="obs")
    parser.add_argument(
        "--load-raw-weights", action="store_true", help="Load encoder_state_dict, not EMA (avg_*)"
    )
    parser.add_argument("--no-dkd", action="store_true", help="Skip map_gene diagnostic block")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_DKD)
    parser.add_argument("--map-mode", choices=("feature_name", "feature_id"), default="feature_name")
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS, help="0 = all")
    parser.add_argument("--no-raw-layer", action="store_true")
    parser.add_argument("--no-encoder", action="store_true")
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Run subfolder is created under here")
    args = parser.parse_args()

    _ensure_path()
    os.environ.setdefault("OMNICELL_DATA_DIR", "/orcd/scratch/bcs/002/njwfish/Omnicell_datasets/")

    from constants import GENE_MANAGER_PATH
    from generative.utils.loading import load_encoder

    with open(GENE_MANAGER_PATH, "rb") as f:
        gene_manager = pickle.load(f)

    n_sh = _describe_shared_genes(gene_manager)
    need_h5 = args.h5ad.is_file() and (not args.no_dkd or not args.no_encoder)
    data: Optional[DkdEncoderData] = None
    if need_h5:
        used_eg = _load_used_genes_from_checkpoint(args.checkpoint)
        data = load_dkd_encoder_data(
            args.h5ad,
            max(0, args.max_cells),
            not args.no_raw_layer,
            args.map_mode,
            gene_manager,
            used_eg,
        )
    if (not args.no_dkd) and data is not None:
        print_dkd_map_diagnostics(data, gene_manager, args.checkpoint)

    if args.no_encoder:
        return 0

    if data is None:
        print("No h5ad data; encoder only. Exiting (no embeddings).")
        if not args.h5ad.is_file():
            print(f"  h5ad missing: {args.h5ad}")
        return 0

    base_dir = str(PACKAGE_DIR)
    encoder = load_encoder(
        config=args.config,
        base_dir=base_dir,
        gene_manager=gene_manager,
        checkpoint=str(args.checkpoint),
        load_avg=not args.load_raw_weights,
    )
    d_in = _encoder_in_dim(encoder)
    n_p = sum(p.numel() for p in encoder.parameters())
    print("\n--- load_encoder ---")
    print(f"  {type(encoder).__name__}  n_params={n_p}  device={next(encoder.parameters()).device}  in_dim={d_in}")
    if d_in is not None and d_in != n_sh:
        print(f"  WARNING: in_dim {d_in} != len(shared) {n_sh}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_base.expanduser().resolve() / f"enc_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = (
        f"checkpoint={args.checkpoint}\nconfig={args.config}\nh5ad={args.h5ad}\n"
        f"max_cells={args.max_cells}  map_mode={args.map_mode}\n"
    )
    (out_dir / "run_meta.txt").write_text(meta, encoding="utf-8")

    Xs = data.X_shared
    g_d = int(Xs.shape[1])
    n_cells = int(Xs.shape[0])
    print(f"\n--- encode: {n_cells} cells × {g_d} shared genes ---")

    print("\n(1) encode_singleton_sets — (batch,1,G), return_cell_embeddings=False, chunked")
    e_sin = encode_singleton_sets(
        encoder, Xs, batch_size=BATCH_SIZE_ENCODE
    )
    np.save(out_dir / "embeddings_singleton.npy", e_sin)
    print(f"  saved {out_dir / 'embeddings_singleton.npy'}  shape={e_sin.shape}")

    print("\n(2) encode_joint_set_per_cell_readout — (1,N,G), one forward")
    e_jt = encode_joint_set_per_cell_readout(encoder, Xs)
    np.save(out_dir / "embeddings_joint_set.npy", e_jt)
    print(f"  saved {out_dir / 'embeddings_joint_set.npy'}  shape={e_jt.shape}")

    print("\n(3) encode_grouped_by_label — (1, n_ct, G) per cell_type; SetNorm sees one type")
    e_grp = encode_grouped_by_label(encoder, Xs, data.cell_types)
    np.save(out_dir / "embeddings_grouped.npy", e_grp)
    print(f"  saved {out_dir / 'embeddings_grouped.npy'}  shape={e_grp.shape}")

    print("\n(4) PCA baseline on X_shared (reference)")
    e_pca = _pca_baseline(Xs, n_comp=50)
    np.save(out_dir / "embeddings_pca.npy", e_pca)
    print(f"  saved {out_dir / 'embeddings_pca.npy'}  shape={e_pca.shape}")

    kc = int(min(LAYOUT_COMPARE_CELLS, e_sin.shape[0], e_jt.shape[0]))
    print_layout_compare(e_sin, e_jt, kc, g_d)

    print("\n--- UMAPs (scFM_eval style) ---")
    for sub, title, e in [
        ("umap_singleton", "Omnicell enc singleton (B,1,G)", e_sin),
        ("umap_joint", "Omnicell enc joint (1,N,G) readout", e_jt),
        ("umap_grouped", "Omnicell enc grouped-by-label (1,n_ct,G)", e_grp),
        ("umap_pca", "PCA baseline on X_shared", e_pca),
    ]:
        write_umap_figures(out_dir / sub, title, e, data.cell_types, data.batches)

    print("\n--- Batch + bio metrics (scFM_eval-style) ---")
    rows: list[dict[str, Any]] = []
    for tag, e in [
        ("singleton", e_sin),
        ("joint_set", e_jt),
        ("grouped_by_label", e_grp),
        ("pca", e_pca),
    ]:
        print(f"  computing scores for {tag} (shape {e.shape}) ...")
        r = compute_embedding_scores(tag, e, data.cell_types, data.batches)
        print(
            f"    ASW_label={r.get('ASW_label', float('nan')):.4f}  "
            f"ASW_batch={r.get('ASW_batch', float('nan')):.4f}  "
            f"iLISI_med={r.get('iLISI_med', float('nan')):.4f}  "
            f"cLISI_med={r.get('cLISI_med', float('nan')):.4f}"
        )
        rows.append(r)
    write_metrics_csv(out_dir / "metrics.csv", rows)

    print(f"\nAll outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
