#!/usr/bin/env python3
"""Direct Omnicell embedding test — bypasses the extractor entirely.

Loads the DKD dataset, swaps to raw counts, sets up sys.path for cell-types,
imports OmnicellBase directly, runs map_genes + encoder, and produces UMAPs.

This is the "trust nothing, verify everything" version: every step is logged
so we can confirm gene format, normalization, mapping counts, and embedding
quality match what cell-types expects.

Usage (run from an env that has torch, anndata, scanpy, matplotlib)::

    cd /orcd/data/omarabu/001/rconci/scFM_eval
    export OMNICELL_DATA_DIR=/orcd/data/omarabu/001/Omnicell_datasets/
    python debug_omnicell_direct.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_OUT_BASE = _REPO_ROOT / "debug_omnicell_umap"

log = logging.getLogger("direct_omnicell")

# ── Configuration ─────────────────────────────────────────────────────────────
H5AD_PATH = Path("/orcd/data/omarabu/001/Omnicell_datasets/bio_batch_eval_data/dkd.h5ad")
CHECKPOINT = Path(
    "/orcd/data/omarabu/001/njwfish/cell-types/cell_types/outputs/obs/"
    "517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_5.pt"
)
CELL_TYPES_ROOT = Path("/orcd/data/omarabu/001/rconci/cell-types")
PACKAGE_DIR = CELL_TYPES_ROOT / "cell_types"
GLOBAL_GENE_LIST = Path(
    "/orcd/data/omarabu/001/Omnicell_datasets/protocol_embeddings/genes/global_gene_mapping.parquet"
)
BIO_COL = "cell_type"
BATCH_COL = "assay"
MAX_CELLS = 2000
BATCH_SIZE = 4096
GENE_TYPES = "feature_name"


def _configure_logging(out_dir: Path) -> None:
    """Log to console and ``out_dir/run.log`` (line-buffered file for live tail)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    log.setLevel(logging.INFO)
    log.propagate = True


def _write_timing_report(
    out_dir: Path,
    markers: list[tuple[str, float]],
    wall_total_s: float,
) -> None:
    """Write ``timing.txt`` with cumulative and per-phase deltas."""
    lines = [
        f"debug_omnicell_direct.py — wall-clock total: {wall_total_s:.2f}s",
        "",
        "phase (cumulative seconds from script start)",
        "-" * 60,
    ]
    prev = 0.0
    for name, t in markers:
        delta = t - prev
        lines.append(f"  {name:40s}  cum {t:10.2f}s  Δ {delta:8.2f}s")
        prev = t
    lines.append("-" * 60)
    lines.append(f"  {'TOTAL':40s}  {wall_total_s:10.2f}s")
    text = "\n".join(lines) + "\n"
    (out_dir / "timing.txt").write_text(text, encoding="utf-8")
    log.info("Wrote timing report: %s", out_dir / "timing.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=_DEFAULT_OUT_BASE,
        help="Parent directory; a timestamped run folder is created inside.",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_base.expanduser().resolve() / f"direct_{run_id}")
    _configure_logging(out_dir)

    wall0 = time.perf_counter()
    markers: list[tuple[str, float]] = []

    def tick(label: str) -> None:
        markers.append((label, time.perf_counter() - wall0))
        log.info("TIMING +%.2fs cumulative — %s", markers[-1][1], label)

    tick("run_start")

    # ── 1. Set up sys.path exactly like run_omnicell_embed.py ─────────────
    for p in [
        str(CELL_TYPES_ROOT),
        str(PACKAGE_DIR),
        str(PACKAGE_DIR / "bulk_prediction"),
        str(PACKAGE_DIR / "generative"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import os
    os.environ.setdefault("OMNICELL_DATA_DIR", "/orcd/data/omarabu/001/Omnicell_datasets/")

    # ── 2. Load data ──────────────────────────────────────────────────────
    log.info("Loading %s ...", H5AD_PATH)
    t0 = time.perf_counter()
    adata = ad.read_h5ad(H5AD_PATH)
    if hasattr(adata, "isbacked") and adata.isbacked:
        adata = adata.to_memory()
    log.info("Loaded in %.1fs: %d cells × %d genes", time.perf_counter() - t0, adata.n_obs, adata.n_vars)
    tick("h5ad_loaded")

    # ── 3. Swap to raw counts ─────────────────────────────────────────────
    X_orig = adata.X
    if issparse(X_orig):
        sample = X_orig.data[:5000]
    else:
        sample = np.asarray(X_orig).ravel()[:5000]
    int_frac = float(np.mean(np.equal(np.mod(sample, 1), 0))) if len(sample) > 0 else 0
    log.info(
        "BEFORE raw swap: X dtype=%s  range=[%.4f, %.4f]  int_frac=%.1f%%",
        X_orig.dtype, float(np.min(sample)), float(np.max(sample)), int_frac * 100,
    )

    if adata.raw is not None:
        log.info("adata.raw exists — swapping to raw counts (adata.raw.to_adata())")
        adata = adata.raw.to_adata()
    else:
        log.warning("adata.raw is None — data may not be raw counts!")

    X_raw = adata.X
    if issparse(X_raw):
        sample_raw = X_raw.data[:5000]
    else:
        sample_raw = np.asarray(X_raw).ravel()[:5000]
    int_frac_raw = float(np.mean(np.equal(np.mod(sample_raw, 1), 0))) if len(sample_raw) > 0 else 0
    log.info(
        "AFTER raw swap: X dtype=%s  range=[%.4f, %.4f]  int_frac=%.1f%%  → %s",
        X_raw.dtype, float(np.min(sample_raw)), float(np.max(sample_raw)),
        int_frac_raw * 100,
        "RAW COUNTS" if int_frac_raw > 0.95 else "WARNING: NOT RAW",
    )

    # ── 4. Gene resolution (exactly what extract.py does) ─────────────────
    log.info("var.index[:5] = %s  (should be Ensembl IDs)", adata.var_names[:5].tolist())
    log.info("var.columns = %s", list(adata.var.columns))

    if GENE_TYPES == "feature_name":
        if "gene_symbol" in adata.var.columns:
            source = "gene_symbol"
            adata.var.index = adata.var["gene_symbol"].astype(str).values
        elif "feature_name" in adata.var.columns:
            source = "feature_name"
            adata.var.index = adata.var["feature_name"].astype(str).values
        else:
            source = "var.index (no feature_name or gene_symbol)"
    else:
        if "gene_id" in adata.var.columns:
            source = "gene_id"
            adata.var.index = adata.var["gene_id"].astype(str).values
        elif "feature_id" in adata.var.columns:
            source = "feature_id"
            adata.var.index = adata.var["feature_id"].astype(str).values
        else:
            source = "var.index (no feature_id or gene_id)"

    log.info("Gene names resolved from: %s", source)
    log.info("var.index[:10] after resolution = %s", adata.var_names[:10].tolist())

    # Quick check against global gene list
    glist = pd.read_parquet(GLOBAL_GENE_LIST)
    global_names = set(glist["feature_name"].values)
    matched = sum(1 for g in adata.var_names if g in global_names)
    ensembl_in_names = sum(1 for g in adata.var_names if str(g).startswith("ENSG"))
    log.info(
        "Gene mapping preview: %d / %d (%.1f%%) match global feature_name;  "
        "%d are Ensembl IDs that won't match",
        matched, adata.n_vars, 100 * matched / max(adata.n_vars, 1), ensembl_in_names,
    )

    # ── 5. Subsample ──────────────────────────────────────────────────────
    if MAX_CELLS and adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(adata.n_obs, size=MAX_CELLS, replace=False))
        adata = adata[idx].copy()
        log.info("Subsampled to %d cells", adata.n_obs)

    cell_types = adata.obs[BIO_COL].to_numpy()
    batches = adata.obs[BATCH_COL].to_numpy()
    tick("adata_ready_raw_genes_subsampled")

    # ── 6. Load ENCODER ONLY (skip generator — not needed for embeddings) ─
    # OmnicellBase.__init__ loads both encoder + generator when config is set.
    # The generator load is very slow (Hydra + full counting-flows model) and
    # completely unnecessary for cell embeddings.  We replicate only the
    # encoder-loading path from generative/utils/loading.py::load_encoder().
    import pickle
    from constants import GENE_MANAGER_PATH

    log.info("Loading gene_manager from %s ...", GENE_MANAGER_PATH)
    with open(GENE_MANAGER_PATH, "rb") as f:
        gene_manager = pickle.load(f)
    log.info(
        "gene_manager: %d total_genes, %d shared_genes",
        len(gene_manager["total_genes"]), len(gene_manager["shared_genes"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load checkpoint ONCE (it's a huge file on NFS — avoid loading twice) ──
    log.info("Loading checkpoint (single load): %s", CHECKPOINT)
    t_ckpt = time.perf_counter()
    ckpt_data = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    log.info("Checkpoint loaded in %.1fs", time.perf_counter() - t_ckpt)
    tick("checkpoint_torch_loaded")
    log.info("Checkpoint top-level keys: %s", list(ckpt_data.keys()))

    used_genes = ckpt_data.get("used_genes")
    log.info(
        "used_genes: %s (count=%s)",
        type(used_genes).__name__,
        len(used_genes) if used_genes is not None else "None",
    )

    # ── Build encoder from Hydra config (no weights yet) ──────────────────
    log.info("Building encoder architecture from Hydra config ...")
    t_build = time.perf_counter()

    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    gen_dir = str(PACKAGE_DIR / "generative")
    os.chdir(gen_dir)
    GlobalHydra.instance().clear()
    config_dir = os.path.join(gen_dir, "config")

    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name="obs")
    OmegaConf.resolve(cfg)

    cfg.encoder.in_dim = len(gene_manager["shared_genes"])
    cfg.encoder.linear_out_dim = len(gene_manager["total_genes"])
    log.info("Hydra config ready in %.1fs", time.perf_counter() - t_build)

    # ── Build TWO encoders (EMA + raw) from same checkpoint dict ────────
    def _build_encoder():
        """Instantiate a fresh encoder from Hydra config (no weights)."""
        enc = hydra.utils.instantiate(OmegaConf.to_container(cfg.encoder))
        return enc

    # EMA encoder
    log.info("Loading EMA (averaged) encoder weights ...")
    t_w = time.perf_counter()
    encoder_ema = _build_encoder()
    avg_wrapper = hydra.utils.instantiate(
        OmegaConf.to_container(cfg.averaging), model=encoder_ema
    )
    avg_wrapper.load_state_dict(ckpt_data["avg_encoder_state_dict"], strict=True)
    encoder_ema.load_state_dict(avg_wrapper.module.state_dict())
    del avg_wrapper
    encoder_ema.to(device).eval()
    log.info("EMA encoder ready in %.1fs", time.perf_counter() - t_w)

    # Raw encoder
    log.info("Loading raw encoder weights ...")
    t_w = time.perf_counter()
    encoder_raw = _build_encoder()
    encoder_raw.load_state_dict(ckpt_data["encoder_state_dict"], strict=True)
    encoder_raw.to(device).eval()
    log.info("Raw encoder ready in %.1fs", time.perf_counter() - t_w)

    del ckpt_data
    log.info("Checkpoint dict freed. Total load+build time: %.1fs", time.perf_counter() - t_ckpt)
    tick("encoders_ema_and_raw_on_device")

    # ── 8. map_genes — using the standalone function directly ─────────────
    from tasks.data.genes import map_genes

    log.info("Calling map_genes(adata, %r, gene_manager, used_genes=...) ...", GENE_TYPES)
    t_map = time.perf_counter()
    mapped_data = map_genes(adata, GENE_TYPES, gene_manager, used_genes=used_genes)
    adata_mapped = mapped_data["adata"]
    shared_gene_ids = mapped_data["shared_gene_ids"]
    total_gene_ids = mapped_data["total_gene_ids"]
    gene_names = mapped_data["gene_names"]
    log.info("map_genes done in %.1fs", time.perf_counter() - t_map)
    tick("map_genes_done")
    log.info("Mapped adata: %d cells × %d genes", adata_mapped.n_obs, adata_mapped.n_vars)
    log.info("total_gene_ids count: %d", len(total_gene_ids))
    log.info("shared_gene_ids count: %d", len(shared_gene_ids))
    log.info("gene_names sample: %s", list(gene_names[:10]))

    X_mapped = adata_mapped.X
    if issparse(X_mapped):
        X_mapped = X_mapped.toarray()
    X_mapped = np.asarray(X_mapped, dtype=np.float32)
    log.info("X_mapped shape: %s  dtype: %s", X_mapped.shape, X_mapped.dtype)
    log.info(
        "X_mapped stats: min=%.4f  max=%.4f  mean=%.4f  nnz_frac=%.4f",
        np.min(X_mapped), np.max(X_mapped), np.mean(X_mapped),
        np.count_nonzero(X_mapped) / max(X_mapped.size, 1),
    )

    X_shared = X_mapped[:, shared_gene_ids].astype(np.float32)
    log.info("X_shared shape: %s", X_shared.shape)
    log.info(
        "X_shared stats: min=%.4f  max=%.4f  mean=%.4f  nnz_frac=%.4f",
        np.min(X_shared), np.max(X_shared), np.mean(X_shared),
        np.count_nonzero(X_shared) / max(X_shared.size, 1),
    )

    # ── 9. Run BOTH encoders and compare ──────────────────────────────────
    def _embed(encoder, name: str) -> np.ndarray:
        log.info("Running %s encoder on %d cells ...", name, X_shared.shape[0])
        emb_list: list[np.ndarray] = []
        n_cells = X_shared.shape[0]
        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
        t0 = time.perf_counter()
        for start in range(0, n_cells, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_cells)
            x_t = torch.from_numpy(X_shared[start:end]).to(device).unsqueeze(1)
            with torch.no_grad():
                _, cell_emb, _ = encoder(x_t, return_cell_embeddings=True)
            emb_list.append(cell_emb.squeeze(1).cpu().numpy())
        emb = np.concatenate(emb_list, axis=0).astype(np.float32)
        log.info(
            "%s embeddings: shape %s  time %.1fs  "
            "min=%.4f max=%.4f mean=%.4f std=%.4f  nan=%s inf=%s",
            name, emb.shape, time.perf_counter() - t0,
            np.min(emb), np.max(emb), np.mean(emb), np.std(emb),
            np.any(np.isnan(emb)), np.any(np.isinf(emb)),
        )
        zero_rows = int(np.sum(np.all(emb == 0, axis=1)))
        const_rows = int(np.sum(np.std(emb, axis=1) < 1e-8))
        if zero_rows > 0 or const_rows > 0:
            log.warning("%s: all-zero=%d  near-constant=%d", name, zero_rows, const_rows)
        return emb

    emb_ema = _embed(encoder_ema, "EMA")
    emb_raw = _embed(encoder_raw, "RAW")

    # Compare the two
    diff = np.abs(emb_ema - emb_raw)
    cos_sims = np.sum(emb_ema * emb_raw, axis=1) / (
        np.linalg.norm(emb_ema, axis=1) * np.linalg.norm(emb_raw, axis=1) + 1e-12
    )
    log.info(
        "EMA vs RAW: mean_abs_diff=%.4f  max_abs_diff=%.4f  "
        "cosine_sim mean=%.4f min=%.4f",
        np.mean(diff), np.max(diff), np.mean(cos_sims), np.min(cos_sims),
    )
    tick("forward_pass_ema_and_raw_done")

    # ── 10. Save + UMAPs for BOTH ─────────────────────────────────────────
    meta = (
        f"h5ad={H5AD_PATH}\n"
        f"checkpoint={CHECKPOINT}\n"
        f"max_cells={MAX_CELLS}\n"
        f"run_id={run_id}\n"
    )
    (out_dir / "run_meta.txt").write_text(meta, encoding="utf-8")

    for emb, tag in [(emb_ema, "ema"), (emb_raw, "raw")]:
        np.save(out_dir / f"embeddings_{tag}.npy", emb)
        log.info("Saved %s embeddings to %s", tag, out_dir / f"embeddings_{tag}.npy")

        log.info("Computing UMAP for %s ...", tag)
        tmp = ad.AnnData(X=emb)
        tmp.obs["cell_type"] = cell_types.astype(str)
        tmp.obs["batch"] = batches.astype(str)
        n_comp = min(50, emb.shape[1], emb.shape[0] - 1)
        sc.tl.pca(tmp, n_comps=n_comp, svd_solver="arpack")
        sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=n_comp, use_rep="X_pca")
        sc.tl.umap(tmp, min_dist=0.3)
        u = tmp.obsm["X_umap"]

        for col, label in [("cell_type", BIO_COL), ("batch", BATCH_COL)]:
            fig, ax = plt.subplots(figsize=(10, 7))
            cats = sorted(tmp.obs[col].unique())
            cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
            lut = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
            colors = [lut[c] for c in tmp.obs[col]]
            ax.scatter(u[:, 0], u[:, 1], c=colors, s=3, alpha=0.8, linewidths=0, rasterized=True)
            ax.set_title(f"Omnicell {tag.upper()} — DKD — {label}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=c,
                           markerfacecolor=lut[c], markersize=6)
                for c in cats[:20]
            ]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
            fig.tight_layout()
            out_path = out_dir / f"umap_{tag}_{col}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Wrote %s", out_path)

    tick("umaps_saved")
    wall_total = time.perf_counter() - wall0
    _write_timing_report(out_dir, markers, wall_total)
    log.info("All done. Outputs: %s", out_dir)
    log.info("Log file: %s", out_dir / "run.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
