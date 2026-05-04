#!/usr/bin/env python3
"""Pre-flight inspector for Omnicell evaluation datasets.

For each input ``.h5ad`` file, reports — without running the encoder — exactly
what the Omnicell pipeline will see:

* RAW counts available? (``adata.raw`` present, integer dtype, max/min, integer fraction)
* Currently-detected ``DataState`` (``utils.data_state.get_data_state``) for both
  ``adata.X`` and ``adata.raw`` — so log1p / TP10K / counts mismatches show up
  before the sweep launches.
* Gene identifiers in ``var``: Ensembl-vs-symbol on ``var.index``, presence of
  ``gene_id`` / ``gene_symbol`` / ``feature_id`` / ``feature_name`` columns.
* **Mapping coverage against the cell-types global gene list** (parquet at
  ``$OMNICELL_DATA_DIR/protocol_embeddings/genes/global_gene_mapping.parquet``).
  This is the same mapping ``cell_types/tasks/data/genes.py::map_genes_to_global_list``
  performs at run time.
* **Coverage after the ``used_genes`` filter** (loaded from ``--checkpoint`` or
  ``--gene-manager``). This is the actual ``shared_gene_ids`` slice the encoder
  will receive.

Outputs:
    - One row per dataset to stdout (human-readable).
    - ``--csv PATH`` → machine-readable summary across all datasets.
    - ``--json-dir DIR`` → one JSON per dataset for sweep tooling.

Examples:
    # Single file
    python inspect_omnicell_inputs.py \\
        --h5ad /orcd/scratch/.../dkd.h5ad \\
        --checkpoint /orcd/scratch/.../checkpoint_epoch_5.pt

    # Whole task cohort, with CSV + JSON outputs
    python inspect_omnicell_inputs.py \
        --h5ad-dir /orcd/scratch/bcs/002/njwfish/Omnicell_datasets/bio_batch_eval_data \
        --checkpoint /orcd/scratch/bcs/002/njwfish/cell-types/cell_types/outputs/obs/517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_5.pt \
        --csv preflight.csv \
        --json-dir preflight_json

This script imports from cell-types only to load the gene_manager pickle and the
global gene list parquet. It does NOT instantiate the encoder, so it is safe to
run in any env that has anndata + scanpy + numpy + pandas.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# anndata>=0.10 probes ``cupy.is_cupy_importable()`` at import time. Inside cupy,
# ``_get_conda_cuda_path`` calls ``os.path.join(os.environ.get("CONDA_PREFIX"), ...)``
# which raises ``TypeError`` when CONDA_PREFIX is unset (e.g. fresh non-interactive
# ssh login, ``srun``, or any environment that didn't run ``conda activate``).
# Default it to the running interpreter's prefix so the import succeeds; we don't
# need cupy here, only anndata.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

# Defaults match cell_types/constants.py so this script is self-contained.
_DEFAULT_OMNICELL_DATA_DIR = "/orcd/scratch/bcs/002/njwfish/Omnicell_datasets/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("preflight")


# ────────────────────────────────────────────────────────────────────────────
# DataState — local copy of the runtime heuristics so we don't import the
# scFM_eval package (avoids dragging in scanpy.tl etc. at preflight time).
# Kept in lockstep with utils/data_state.py::_detect_from_distribution.
# ────────────────────────────────────────────────────────────────────────────


def _data_state_from_x(x_sample: np.ndarray) -> str:
    """Return ``raw`` / ``normalized`` / ``log1p`` / ``unknown`` from a sample."""
    try:
        if x_sample.size == 0:
            return "unknown"
        max_val = float(np.max(x_sample))
        if max_val < 25:
            return "log1p"
        # Integer-ness check on a smaller subset (avoids quadratic memory).
        sub = x_sample[: min(1000, x_sample.shape[0])] if x_sample.ndim else x_sample
        is_int = bool(np.allclose(sub, np.round(sub), rtol=1e-5))
        if max_val > 100 and is_int:
            return "raw"
        if max_val < 100:
            return "normalized"
        return "raw"
    except Exception:
        return "unknown"


# ────────────────────────────────────────────────────────────────────────────
# Per-dataset report
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetReport:
    """Everything the omnicell pipeline cares about for one h5ad file."""

    path: str
    n_obs: int = 0
    n_vars: int = 0

    # X / RAW state
    has_raw: bool = False
    x_dtype: str = ""
    x_state: str = "unknown"
    x_max: float = float("nan")
    x_min: float = float("nan")
    x_int_frac: float = float("nan")
    raw_dtype: str = ""
    raw_state: str = "unknown"
    raw_max: float = float("nan")
    raw_min: float = float("nan")
    raw_int_frac: float = float("nan")
    declared_data_state: Optional[str] = None  # adata.uns['data_state'] if any
    has_uns_log1p: bool = False
    counts_layer_name: Optional[str] = None  # 'counts' / 'original_X' / etc.

    # Gene identifiers
    var_index_type: str = "unknown"  # 'ensembl' / 'symbol' / 'mixed'
    n_ensg_in_index: int = 0
    has_col_gene_id: bool = False
    has_col_gene_symbol: bool = False
    has_col_feature_id: bool = False
    has_col_feature_name: bool = False

    # Global-gene-list mapping
    map_mode_used: str = "feature_name"
    map_source: str = ""  # which column was used as the lookup
    n_genes_for_lookup: int = 0
    n_mapped_to_global: int = 0
    pct_mapped_to_global: float = float("nan")

    # used_genes overlap
    used_genes_loaded_from: Optional[str] = None
    n_used_genes_total: int = 0
    n_mapped_in_used_genes: int = 0
    pct_kept_after_used_genes: float = float("nan")

    # shared-genes overlap (encoder input width)
    n_shared_genes_total: int = 0
    n_mapped_in_shared_genes: int = 0
    pct_shared_covered: float = float("nan")

    warnings: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _sample_x(matrix: Any, n_rows: int = 1000) -> np.ndarray:
    """Return up to ``n_rows`` rows of ``matrix`` as a dense float64 array."""
    n = matrix.shape[0]
    head = matrix[: min(n_rows, n)]
    if issparse(head):
        head = head.toarray()
    return np.asarray(head, dtype=np.float64)


def _value_summary(arr: np.ndarray) -> tuple[float, float, float]:
    """Return ``(max, min, integer_fraction)`` of nonzero entries."""
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    flat = arr.ravel()
    nz = flat[flat != 0]
    sample = nz[: min(50_000, nz.size)] if nz.size else flat[:1]
    int_frac = (
        float(np.mean(np.isclose(sample, np.rint(sample), atol=1e-5)))
        if sample.size
        else float("nan")
    )
    return float(np.max(arr)), float(np.min(arr)), int_frac


def _detect_var_index_type(var_index: pd.Index) -> tuple[str, int]:
    """Heuristic Ensembl vs symbol on ``var.index``."""
    sample = var_index[: min(500, len(var_index))].astype(str)
    n_ensg = int(sum(1 for s in sample if s.startswith(("ENSG", "ENSMUSG"))))
    if n_ensg > 0.8 * len(sample):
        return "ensembl", n_ensg
    if n_ensg < 0.05 * len(sample):
        return "symbol", n_ensg
    return "mixed", n_ensg


def _normalize_for_lookup(genes: Iterable[str]) -> np.ndarray:
    """Strip ``"SYMBOL (ENTREZ)"`` to match ``normalize_gene_name_for_vocab``.

    NOTE: production's ``map_genes_to_global_list`` is case-sensitive — it does
    a plain dict ``get`` against the global gene list's ``feature_name`` column.
    We must *not* uppercase here, otherwise we'll over-report mapping coverage
    for any case-mismatched dataset (e.g. mouse 'Tspan6' vs. human 'TSPAN6').
    """
    out: list[str] = []
    for g in genes:
        s = str(g).strip()
        if " (" in s and s.endswith(")"):
            s = s.split(" (", 1)[0].strip()
        out.append(s)
    return np.asarray(out, dtype=object)


def _load_used_genes(
    checkpoint: Optional[Path],
    gene_manager: dict[str, Any],
) -> tuple[Optional[set[int]], Optional[str]]:
    """Try checkpoint['used_genes']; fall back to gene_manager['used_genes'] if any."""
    if checkpoint and checkpoint.is_file():
        try:
            import torch  # local import keeps preflight cheap if torch missing

            ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            ug = ck.get("used_genes")
            if ug is not None:
                return {int(x) for x in ug}, str(checkpoint)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not read used_genes from checkpoint %s: %s", checkpoint, e)
    if "used_genes" in gene_manager:
        ug = gene_manager["used_genes"]
        return {int(x) for x in ug}, "gene_manager['used_genes']"
    return None, None


# ────────────────────────────────────────────────────────────────────────────
# Per-dataset inspection
# ────────────────────────────────────────────────────────────────────────────


def inspect_one(
    h5ad_path: Path,
    *,
    global_df: pd.DataFrame,
    gene_manager: dict[str, Any],
    used_genes: Optional[set[int]],
    used_genes_source: Optional[str],
    map_mode: str,
    sample_n: int,
) -> DatasetReport:
    """Build a :class:`DatasetReport` for a single ``.h5ad`` file."""
    report = DatasetReport(path=str(h5ad_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # backed='r' avoids loading X for big files; we only sample rows.
        adata = ad.read_h5ad(h5ad_path, backed="r")

    try:
        report.n_obs, report.n_vars = int(adata.n_obs), int(adata.n_vars)
        report.has_raw = adata.raw is not None
        report.has_uns_log1p = "log1p" in (adata.uns or {})
        report.declared_data_state = (
            str(adata.uns["data_state"]) if "data_state" in (adata.uns or {}) else None
        )
        # Counts layer name (loader copies X into 'counts' or layer aliasing).
        if hasattr(adata, "layers") and adata.layers is not None:
            for name in ("counts", "original_X"):
                if name in adata.layers:
                    report.counts_layer_name = name
                    break

        # ── X stats ──────────────────────────────────────────────────────
        x_sample = _sample_x(adata.X, n_rows=sample_n)
        report.x_dtype = str(adata.X.dtype)
        report.x_max, report.x_min, report.x_int_frac = _value_summary(x_sample)
        report.x_state = report.declared_data_state or _data_state_from_x(x_sample)

        # ── RAW stats ────────────────────────────────────────────────────
        if report.has_raw:
            raw_sample = _sample_x(adata.raw.X, n_rows=sample_n)
            report.raw_dtype = str(adata.raw.X.dtype)
            report.raw_max, report.raw_min, report.raw_int_frac = _value_summary(raw_sample)
            report.raw_state = _data_state_from_x(raw_sample)

        # ── Gene identifiers in var ─────────────────────────────────────
        var_df = adata.var
        report.has_col_gene_id = "gene_id" in var_df.columns
        report.has_col_gene_symbol = "gene_symbol" in var_df.columns
        report.has_col_feature_id = "feature_id" in var_df.columns
        report.has_col_feature_name = "feature_name" in var_df.columns
        report.var_index_type, report.n_ensg_in_index = _detect_var_index_type(var_df.index)

        # ── Pick the column that the runtime would use, in priority order
        # (matches OmnicellExtractor.extract_embeddings + ensure_both_gene_identifiers)
        report.map_mode_used = map_mode
        if map_mode == "feature_name":
            if "gene_symbol" in var_df.columns:
                lookup_genes = var_df["gene_symbol"].astype(str).to_numpy()
                report.map_source = "gene_symbol"
            elif "feature_name" in var_df.columns:
                lookup_genes = var_df["feature_name"].astype(str).to_numpy()
                report.map_source = "feature_name"
            else:
                lookup_genes = var_df.index.astype(str).to_numpy()
                report.map_source = "var.index"
        else:  # feature_id
            if "gene_id" in var_df.columns:
                lookup_genes = var_df["gene_id"].astype(str).to_numpy()
                report.map_source = "gene_id"
            elif "feature_id" in var_df.columns:
                lookup_genes = var_df["feature_id"].astype(str).to_numpy()
                report.map_source = "feature_id"
            else:
                lookup_genes = var_df.index.astype(str).to_numpy()
                report.map_source = "var.index"
        report.n_genes_for_lookup = int(len(lookup_genes))

        # ── Map to global gene list ─────────────────────────────────────
        # Match production semantics exactly (cell_types/scripts/prep/generate_gene_info.py
        # ::map_genes_to_global_list): build {gene -> first idx} from global_df[mode],
        # case-sensitive, no canonicalization. The runtime calls
        # ``normalize_gene_name_for_vocab`` (strip 'SYMBOL (ID)') on var.index
        # before mapping — _normalize_for_lookup does the same for feature_name.
        if map_mode == "feature_name":
            queries = _normalize_for_lookup(lookup_genes)
        else:
            queries = np.asarray([str(g).strip() for g in lookup_genes], dtype=object)
        global_keys = global_df[map_mode].astype(str)

        # Keep first occurrence (matches the runtime's ``drop_duplicates`` semantics
        # in ``cell_types/tasks/data/genes.py::map_genes_to_global_list``).
        gene_to_idx: dict[str, int] = {}
        for k, i in zip(global_keys, global_df.index):
            gene_to_idx.setdefault(k, int(i))

        mapped = np.fromiter(
            (gene_to_idx.get(q, -1) for q in queries), dtype=np.int64, count=len(queries)
        )
        ok = mapped[mapped != -1]
        report.n_mapped_to_global = int(ok.size)
        report.pct_mapped_to_global = (
            round(100.0 * ok.size / len(queries), 2) if len(queries) else 0.0
        )

        # ── used_genes filter ───────────────────────────────────────────
        if used_genes is not None:
            report.used_genes_loaded_from = used_genes_source
            report.n_used_genes_total = len(used_genes)
            keep_mask = np.isin(ok, np.fromiter(used_genes, dtype=np.int64))
            report.n_mapped_in_used_genes = int(keep_mask.sum())
            report.pct_kept_after_used_genes = (
                round(100.0 * keep_mask.sum() / ok.size, 2) if ok.size else 0.0
            )

        # ── shared_genes coverage ───────────────────────────────────────
        shared = np.fromiter(
            (int(g) for g in gene_manager.get("shared_genes", [])),
            dtype=np.int64,
        )
        report.n_shared_genes_total = int(shared.size)
        if shared.size and ok.size:
            shared_in_data = np.isin(shared, ok)
            report.n_mapped_in_shared_genes = int(shared_in_data.sum())
            report.pct_shared_covered = round(
                100.0 * shared_in_data.sum() / shared.size, 2
            )

        # ── Heuristic warnings ──────────────────────────────────────────
        if not report.has_raw and report.x_state != "raw":
            report.warnings.append(
                f"X looks {report.x_state!r} and adata.raw is missing; "
                "Omnicell would fall back to recover_counts_from_normalized "
                "(approximation, not real counts)"
            )
        if report.has_raw and report.raw_state != "raw":
            report.warnings.append(
                f"adata.raw exists but its values look {report.raw_state!r}, not raw counts"
            )
        if report.pct_mapped_to_global < 50.0:
            report.warnings.append(
                f"Only {report.pct_mapped_to_global:.1f}% of input genes mapped to the "
                f"global gene list (lookup column = {report.map_source!r}, "
                f"mode = {map_mode!r}). Likely a gene-name format mismatch."
            )
        if (
            used_genes is not None
            and report.pct_kept_after_used_genes < 50.0
            and report.n_mapped_to_global
        ):
            report.warnings.append(
                f"After used_genes filter, only "
                f"{report.pct_kept_after_used_genes:.1f}% of mapped genes remain"
            )
        if report.pct_shared_covered < 80.0 and report.n_shared_genes_total:
            report.warnings.append(
                f"Only {report.pct_shared_covered:.1f}% of the model's shared_genes "
                f"are present in this dataset — encoder will see lots of zeros."
            )
    finally:
        try:
            adata.file.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def _resolve_global_gene_list(arg: Optional[Path]) -> Path:
    if arg is not None:
        return arg
    data_dir = os.environ.get("OMNICELL_DATA_DIR", _DEFAULT_OMNICELL_DATA_DIR)
    return Path(data_dir) / "protocol_embeddings" / "genes" / "global_gene_mapping.parquet"


def _resolve_gene_manager(arg: Optional[Path]) -> Path:
    if arg is not None:
        return arg
    data_dir = os.environ.get("OMNICELL_DATA_DIR", _DEFAULT_OMNICELL_DATA_DIR)
    return Path(data_dir) / "protocol_embeddings" / "genes" / "gene_manager.pkl"


def _format_human(report: DatasetReport) -> str:
    """One human-readable block per dataset."""
    lines = [
        f"\n=== {report.path}",
        f"  shape         : ({report.n_obs}, {report.n_vars})",
        f"  X             : dtype={report.x_dtype}  state={report.x_state}  "
        f"min={report.x_min:.4g}  max={report.x_max:.4g}  int_frac={report.x_int_frac:.3f}",
    ]
    if report.has_raw:
        lines.append(
            f"  RAW           : dtype={report.raw_dtype}  state={report.raw_state}  "
            f"min={report.raw_min:.4g}  max={report.raw_max:.4g}  int_frac={report.raw_int_frac:.3f}"
        )
    else:
        lines.append("  RAW           : MISSING (adata.raw is None)")
    if report.declared_data_state:
        lines.append(f"  uns.data_state: {report.declared_data_state}")
    if report.has_uns_log1p:
        lines.append("  uns.log1p     : present (scanpy log1p marker)")
    if report.counts_layer_name:
        lines.append(f"  layer         : {report.counts_layer_name!r}")
    lines.append(
        f"  var.index     : {report.var_index_type}  "
        f"({report.n_ensg_in_index} ENSG-like in first 500)"
    )
    cols = [
        c
        for c, has in [
            ("gene_id", report.has_col_gene_id),
            ("gene_symbol", report.has_col_gene_symbol),
            ("feature_id", report.has_col_feature_id),
            ("feature_name", report.has_col_feature_name),
        ]
        if has
    ]
    lines.append(f"  var columns   : {cols}")
    lines.append(
        f"  global mapping: source={report.map_source!r} mode={report.map_mode_used!r} "
        f"→ {report.n_mapped_to_global}/{report.n_genes_for_lookup} "
        f"({report.pct_mapped_to_global}%)"
    )
    if report.used_genes_loaded_from:
        lines.append(
            f"  used_genes    : kept {report.n_mapped_in_used_genes}/{report.n_mapped_to_global} "
            f"({report.pct_kept_after_used_genes}%)  "
            f"[total used_genes={report.n_used_genes_total} from {report.used_genes_loaded_from}]"
        )
    if report.n_shared_genes_total:
        lines.append(
            f"  shared_genes  : {report.n_mapped_in_shared_genes}/{report.n_shared_genes_total} "
            f"present ({report.pct_shared_covered}%)"
        )
    for w in report.warnings:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--h5ad",
        type=Path,
        nargs="*",
        default=None,
        help="One or more .h5ad files to inspect.",
    )
    parser.add_argument(
        "--h5ad-dir",
        type=Path,
        default=None,
        help="Directory to glob for *.h5ad files (non-recursive).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Omnicell checkpoint .pt — used to load 'used_genes' for the keep-rate check.",
    )
    parser.add_argument(
        "--global-gene-list",
        type=Path,
        default=None,
        help="Path to global_gene_mapping*.parquet. Defaults to "
        "$OMNICELL_DATA_DIR/protocol_embeddings/genes/global_gene_mapping.parquet.",
    )
    parser.add_argument(
        "--gene-manager",
        type=Path,
        default=None,
        help="Path to gene_manager.pkl. Defaults to "
        "$OMNICELL_DATA_DIR/protocol_embeddings/genes/gene_manager.pkl.",
    )
    parser.add_argument(
        "--map-mode",
        choices=("feature_name", "feature_id"),
        default="feature_name",
        help="Lookup mode (matches yaml gene_types). Default: feature_name.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="Number of rows used to estimate distribution stats (default 2000).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV summary (one row per dataset).",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Optional dir; one JSON sidecar per dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any dataset emits a warning (useful as a CI gate before sweeps).",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.h5ad:
        paths.extend(args.h5ad)
    if args.h5ad_dir:
        paths.extend(sorted(args.h5ad_dir.glob("*.h5ad")))
    if not paths:
        parser.error("Provide at least one --h5ad path or --h5ad-dir.")

    global_gene_list_path = _resolve_global_gene_list(args.global_gene_list)
    gene_manager_path = _resolve_gene_manager(args.gene_manager)
    if not global_gene_list_path.is_file():
        parser.error(f"global gene list not found: {global_gene_list_path}")
    if not gene_manager_path.is_file():
        parser.error(f"gene_manager pickle not found: {gene_manager_path}")

    logger.info("Loading global gene list: %s", global_gene_list_path)
    global_df = pd.read_parquet(global_gene_list_path)
    logger.info("Loading gene manager: %s", gene_manager_path)
    with open(gene_manager_path, "rb") as f:
        gene_manager = pickle.load(f)

    used_genes, used_genes_source = _load_used_genes(args.checkpoint, gene_manager)
    if args.checkpoint and used_genes is None:
        logger.warning(
            "Could not derive used_genes from checkpoint or gene_manager; "
            "the keep-rate column will be empty."
        )

    reports: list[DatasetReport] = []
    for p in paths:
        if not p.is_file():
            logger.warning("Skipping missing file: %s", p)
            continue
        logger.info("Inspecting %s ...", p)
        try:
            r = inspect_one(
                p,
                global_df=global_df,
                gene_manager=gene_manager,
                used_genes=used_genes,
                used_genes_source=used_genes_source,
                map_mode=args.map_mode,
                sample_n=args.sample_rows,
            )
        except Exception as e:
            logger.error("FAILED inspecting %s: %s", p, e)
            continue
        reports.append(r)
        print(_format_human(r))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([asdict(r) for r in reports])
        # Lists do not survive CSV well — drop the 'warnings' column to a joined string.
        if "warnings" in df.columns:
            df["warnings"] = df["warnings"].apply(lambda xs: " | ".join(xs))
        df.to_csv(args.csv, index=False)
        logger.info("Wrote CSV summary to %s", args.csv)

    if args.json_dir:
        args.json_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            stem = Path(r.path).stem
            (args.json_dir / f"{stem}.json").write_text(
                json.dumps(asdict(r), indent=2),
                encoding="utf-8",
            )
        logger.info("Wrote per-dataset JSONs to %s", args.json_dir)

    n_warn = sum(1 for r in reports if r.warnings)
    if n_warn:
        logger.warning("%d/%d datasets emitted warnings.", n_warn, len(reports))
    return 1 if (args.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
