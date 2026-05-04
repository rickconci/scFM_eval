#!/usr/bin/env python
"""Omnicell (cell-types) encoder embedding extraction.

Uses the Omnicell encoder from the cell-types repo to extract cell embeddings.
Requires cell-types to be available (e.g. same env as scfm_meta) and
OMNICELL_DATA_DIR set to a dir containing protocol_embeddings/ (global_gene_mapping_v2_new.parquet,
gene_manager_serialized.pkl); the cell-types package reads these at import.
For best gene matching, run ensure_both_gene_identifiers(normalize=True) on
the loader before extraction so adata.var has gene_symbol (and gene_id).

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \\
        --checkpoint_path /path/to/omnicell_checkpoint_epoch27.pt \\
        --base_dir /path/to/cell-types/cell_types
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc

try:
    from anndata import ImplicitModificationWarning
except ImportError:  # pragma: no cover - defensive for odd installs
    ImplicitModificationWarning = type(  # type: ignore[misc,assignment]
        "ImplicitModificationWarning",
        (Warning,),
        {},
    )

# Base extractor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from base_extract import (
    BaseExtractor,
    create_argument_parser,
    normalize_adata_var_gene_names,
    run_extraction,
)

logger = logging.getLogger(__name__)


# Regexes for the lines emitted by ``cell_types/tasks/data/genes.py::map_genes(_to_global_list)``
# and ``run_omnicell_embed.py::_get_cell_embeddings``. Used to scrape per-dataset gene-mapping
# coverage out of the subprocess log so the parent run keeps a paper trail.
_RE_TOTAL = re.compile(r"^\s*Total genes to map:\s*(\d+)", re.MULTILINE)
_RE_MAPPED = re.compile(r"^\s*Successfully mapped:\s*(\d+)", re.MULTILINE)
_RE_REMOVED = re.compile(
    r"^\s*Removing\s+(\d+)\s+genes since they are not in the used genes list",
    re.MULTILINE,
)
_RE_X_SHARED = re.compile(
    r"X_shared shape=\((\d+),\s*(\d+)\)",
    re.MULTILINE,
)


def _summarize_omnicell_subprocess_log(log_path: Path) -> dict[str, Any]:
    """Scrape gene-mapping coverage + encoder input shape out of the subprocess log.

    The omnicell subprocess prints a few self-describing lines that fully characterize
    how the input adata was mapped onto the model vocabulary:

    - ``Total genes to map: N`` / ``Successfully mapped: M`` (pre-``used_genes`` filter)
    - ``Removing K genes since they are not in the used genes list``
    - ``X_shared shape=(N_cells, N_shared)`` (final tensor going into the encoder)

    Returns a dict with whatever it could parse; missing keys mean the corresponding
    line was not in the log (e.g. older run, or subprocess crashed early).
    """
    out: dict[str, Any] = {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    if (m := _RE_TOTAL.search(text)) is not None:
        out["genes_in_input"] = int(m.group(1))
    if (m := _RE_MAPPED.search(text)) is not None:
        out["genes_mapped_to_global"] = int(m.group(1))
    if (m := _RE_REMOVED.search(text)) is not None:
        out["genes_removed_not_in_used_genes"] = int(m.group(1))
    if (m := _RE_X_SHARED.search(text)) is not None:
        out["n_cells"] = int(m.group(1))
        out["n_shared_into_encoder"] = int(m.group(2))

    n_in = out.get("genes_in_input")
    n_map = out.get("genes_mapped_to_global")
    if n_in and n_map is not None:
        out["pct_genes_mapped_to_global"] = round(100.0 * n_map / n_in, 2)
    n_removed = out.get("genes_removed_not_in_used_genes")
    if n_map is not None and n_removed is not None and n_map > 0:
        kept = max(n_map - n_removed, 0)
        out["genes_kept_after_used_genes"] = kept
        out["pct_kept_after_used_genes"] = round(100.0 * kept / n_map, 2)
    return out


class OmnicellExtractor(BaseExtractor):
    """Omnicell (cell-types) encoder extractor.

    Loads the Omnicell encoder + generator from a checkpoint, maps input genes
    to the model's vocabulary (global gene list + used_genes), and returns
    per-cell encoder embeddings.

    **Required input alignment**
    - Gene identifiers are taken from ``adata.var_names`` for mapping to the
      cell-types global gene list (parquet with ``feature_id`` and ``feature_name``).
    - ``gene_types``: ``"feature_name"`` = gene symbols (e.g. HGNC);
      ``"feature_id"`` = Ensembl IDs. Must match the global list column.
    - If ``gene_types == "feature_name"`` and ``adata.var`` has ``gene_symbol``,
      the extractor uses ``gene_symbol`` as the mapping source (recommended:
      run ``ensure_both_gene_identifiers(normalize=True)`` before extraction).
      If ``gene_symbol`` is missing but ``feature_name`` exists (e.g. CELLxGENE
      downloads), ``feature_name`` is used so ``var.index`` is not wrongly
      treated as symbols when it holds Ensembl IDs.
    - If ``gene_types == "feature_id"`` and ``adata.var`` has ``gene_id``,
      the extractor uses ``gene_id`` for mapping; if missing, ``feature_id`` is
      used when present.
    - Gene names like ``"SYMBOL (ENTREZ)"`` are normalized to ``"SYMBOL"``
      before lookup so they match the global list.
    - ``OMNICELL_BASE_DIR``: path to the cell-types repo root (e.g. ``.../cell-types``)
      or the package dir (``.../cell-types/cell_types``). Either works.
    - ``OMNICELL_DATA_DIR`` (required at runtime): path to the dir that contains
      ``protocol_embeddings/`` with ``global_gene_mapping_v2_new.parquet`` and
      ``gene_manager_serialized.pkl``. Set in the environment.
    - ``OMNICELL_PYTHON`` (optional): path to the Python executable to use for the
      embedding subprocess (e.g. your cell-types–specific env). If unset, the
      current process Python is used.

    The extractor runs the cell-types model in a **subprocess** with the correct
    PYTHONPATH and cwd so that cell-types' internal imports work without modifying
    sys.path in the main process or adding __init__.py to the cell-types repo.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config: str = "obs",
        base_dir: str | None = None,
        batch_size: int = 4096,
        device: str = "auto",
        gene_types: str = "feature_name",
        load_avg: bool = True,
        encoding_method: str = "singleton",
        joint_chunk: int = 8192,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Registry calls extractor_class(params); first positional is then the config dict
        if isinstance(checkpoint_path, dict):
            params = checkpoint_path
            checkpoint_path = None
        if params is not None:
            super().__init__(params=params)
            checkpoint_path = self.params.get("checkpoint_path", checkpoint_path)
            config = self.params.get("config", config)
            base_dir = self.params.get("base_dir", base_dir)
            batch_size = self.params.get("batch_size", batch_size)
            device = self.params.get("device", device)
            gene_types = self.params.get("gene_types", gene_types)
            load_avg = self.params.get("load_avg", load_avg)
            encoding_method = self.params.get("encoding_method", encoding_method)
            joint_chunk = self.params.get("joint_chunk", joint_chunk)
        else:
            super().__init__(
                checkpoint_path=checkpoint_path,
                config=config,
                base_dir=base_dir,
                batch_size=batch_size,
                device=device,
                gene_types=gene_types,
                load_avg=load_avg,
                encoding_method=encoding_method,
                joint_chunk=joint_chunk,
                **kwargs,
            )

        if encoding_method not in ("singleton", "joint"):
            raise ValueError(
                f"encoding_method must be 'singleton' or 'joint', got {encoding_method!r}"
            )

        self.checkpoint_path = checkpoint_path
        self.config = config
        self.base_dir = Path(base_dir) if base_dir else None
        self.batch_size = batch_size
        self.device = device
        self.gene_types = gene_types
        self.load_avg = load_avg
        self.encoding_method = encoding_method
        self.joint_chunk = int(joint_chunk)

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Omnicell checkpoint not found: {self.checkpoint_path}"
            )
        if not self.base_dir or not self.base_dir.is_dir():
            raise FileNotFoundError(
                f"Omnicell base_dir (cell_types package) not found: {self.base_dir}"
            )

        logger.info(
            "Omnicell extractor: config=%s, checkpoint=%s, base_dir=%s",
            self.config,
            self.checkpoint_path,
            self.base_dir,
        )

    @property
    def model_name(self) -> str:
        return "omnicell"

    @property
    def embedding_dim(self) -> int:
        return -1  # Known only after extraction (encoder latent_dim)

    def load_model(self) -> None:
        """No-op: model is loaded in the subprocess."""
        pass

    def _repo_paths(self) -> tuple[Path, Path]:
        """Return (repo_root, package_dir) for cell-types."""
        base = self.base_dir.resolve()
        if (base / "cell_types").is_dir():
            return base, base / "cell_types"
        if base.name == "cell_types":
            return base.parent, base
        return base.parent, base

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        repo_root, package_dir = self._repo_paths()
        # Prepare adata in this process (gene alignment + normalization)
        if hasattr(adata, "filename") and adata.isbacked:
            adata = adata.to_memory()
        adata = adata.copy()
        if self.gene_types == "feature_name":
            if "gene_symbol" in adata.var.columns:
                adata.var.index = adata.var["gene_symbol"].astype(str).values
            elif "feature_name" in adata.var.columns:
                adata.var.index = adata.var["feature_name"].astype(str).values
        elif self.gene_types == "feature_id":
            if "gene_id" in adata.var.columns:
                adata.var.index = adata.var["gene_id"].astype(str).values
            elif "feature_id" in adata.var.columns:
                adata.var.index = adata.var["feature_id"].astype(str).values
        normalize_adata_var_gene_names(adata)

        helper = Path(__file__).resolve().parent / "run_omnicell_embed.py"
        env = os.environ.copy()
        bulk_dir = package_dir / "bulk_prediction"
        generative_dir = package_dir / "generative"
        path_parts = [str(repo_root), str(package_dir)]
        if bulk_dir.is_dir():
            path_parts.append(str(bulk_dir))  # so hydra_utils (bulk_prediction/hydra_utils) is found
        if generative_dir.is_dir():
            path_parts.append(str(generative_dir))  # Hydra: generator.* targets live under generative/
        env["PYTHONPATH"] = os.pathsep.join(path_parts + [env.get("PYTHONPATH", "")])
        if "OMNICELL_DATA_DIR" not in env:
            logger.warning("OMNICELL_DATA_DIR not set; cell-types may fail to load gene manager")

        # Persist the subprocess log + a JSON sidecar with the gene-mapping coverage
        # next to the run instead of inside ``/tmp``, so we keep a paper trail of what
        # the encoder actually saw (input → global list → used_genes → encoder shared).
        save_dir_str = self.params.get("save_dir") if hasattr(self, "params") else None
        if save_dir_str:
            persist_dir = Path(save_dir_str) / "extractor_logs"
            persist_dir.mkdir(parents=True, exist_ok=True)
        else:
            persist_dir = None
            logger.warning(
                "OmnicellExtractor: 'save_dir' not in params; subprocess log + gene-coverage "
                "JSON will only live in /tmp and be lost when this run exits."
            )

        with tempfile.TemporaryDirectory(prefix="omnicell_embed_") as tmp:
            input_h5ad = Path(tmp) / "adata.h5ad"
            output_npy = Path(tmp) / "embeddings.npy"
            subprocess_log = Path(tmp) / "omnicell_subprocess.log"
            # Avoid AnnData warning noise on obs/var index coercion when writing temp h5ad.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ImplicitModificationWarning)
                adata.write_h5ad(input_h5ad)

            # Allow overriding the Python used for the subprocess so we can
            # point explicitly to the Omnicell / cell-types env (e.g. MORPH).
            # Falls back to the current interpreter if OMNICELL_PYTHON is unset.
            python_exec = env.get("OMNICELL_PYTHON") or sys.executable

            cmd = [
                python_exec,
                str(helper),
                "--input",
                str(input_h5ad),
                "--output",
                str(output_npy),
                "--checkpoint_path",
                str(self.checkpoint_path),
                "--base_dir",
                str(package_dir),
                "--config",
                self.config,
                "--gene_types",
                self.gene_types,
                "--batch_size",
                str(self.batch_size),
                "--encoding-method",
                self.encoding_method,
                "--joint-chunk",
                str(self.joint_chunk),
            ]
            if not self.load_avg:
                cmd.append("--no-load-avg")
            # Unbuffered child + inherited stdout/stderr so logs appear during long runs
            # (capture_output=True would hide all subprocess output until exit).
            env.setdefault("PYTHONUNBUFFERED", "1")
            logger.info(
                "Starting Omnicell embedding subprocess (%d cells); "
                "progress is written to %s (required when the parent stdout is a pipe, "
                "e.g. parallel_experiments capture_output — otherwise the pipe fills and deadlocks).",
                adata.n_obs,
                subprocess_log,
            )
            # Never inherit stdout/stderr when this process may have stdout/stderr connected
            # to a full PIPE (parallel runner uses subprocess.run(capture_output=True)).
            # The helper prints once per ~10% of batches; ~900k cells → enough lines to fill 64KiB.
            try:
                with open(subprocess_log, "w", encoding="utf-8", errors="replace") as logf:
                    result = subprocess.run(
                        cmd,
                        env=env,
                        cwd=str(repo_root),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                    )
            finally:
                # Always copy the subprocess log out of /tmp before the TemporaryDirectory
                # context tears it down — even if the subprocess crashed.
                if persist_dir is not None and subprocess_log.exists():
                    try:
                        shutil.copy2(subprocess_log, persist_dir / "omnicell_subprocess.log")
                    except OSError as e:  # pragma: no cover - defensive
                        logger.warning("Failed to persist omnicell subprocess log: %s", e)

            # Parse the subprocess log for gene-mapping coverage (works for success
            # and partial-failure runs alike).
            coverage = _summarize_omnicell_subprocess_log(subprocess_log)
            if coverage:
                logger.info(
                    "Omnicell gene coverage | input=%s mapped=%s (%.1f%%) "
                    "used_genes_kept=%s (%.1f%%) shared_into_encoder=%s",
                    coverage.get("genes_in_input", "?"),
                    coverage.get("genes_mapped_to_global", "?"),
                    coverage.get("pct_genes_mapped_to_global", float("nan")),
                    coverage.get("genes_kept_after_used_genes", "?"),
                    coverage.get("pct_kept_after_used_genes", float("nan")),
                    coverage.get("n_shared_into_encoder", "?"),
                )
                if persist_dir is not None:
                    coverage_meta = {
                        **coverage,
                        "checkpoint_path": str(self.checkpoint_path),
                        "config": self.config,
                        "gene_types": self.gene_types,
                        "load_avg": bool(self.load_avg),
                        "encoding_method": self.encoding_method,
                        "batch_size": int(self.batch_size),
                    }
                    try:
                        (persist_dir / "omnicell_gene_coverage.json").write_text(
                            json.dumps(coverage_meta, indent=2),
                            encoding="utf-8",
                        )
                    except OSError as e:  # pragma: no cover - defensive
                        logger.warning("Failed to write omnicell_gene_coverage.json: %s", e)
            else:
                logger.warning(
                    "OmnicellExtractor: could not parse gene-mapping coverage from subprocess log "
                    "(%s). The subprocess may have crashed before mapping or the log format changed.",
                    subprocess_log,
                )

            if result.returncode != 0:
                try:
                    tail = subprocess_log.read_text(encoding="utf-8", errors="replace")[
                        -12000:
                    ]
                except OSError:
                    tail = "(could not read subprocess log)"
                logger.error(
                    "Omnicell subprocess exited with code %s. Log tail:\n%s",
                    result.returncode,
                    tail,
                )
                raise RuntimeError(
                    f"Omnicell embedding subprocess failed with code {result.returncode}"
                )
            embeddings = np.load(output_npy)
        logger.info("Omnicell embeddings shape: %s, dtype %s", embeddings.shape, embeddings.dtype)
        return embeddings.astype(np.float32)


def main() -> None:
    parser = create_argument_parser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Omnicell checkpoint .pt file",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to cell_types package directory (cell-types/cell_types)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="obs",
        help="Cell-types config name (default: obs)",
    )
    parser.add_argument(
        "--gene_types",
        type=str,
        default="feature_name",
        choices=["feature_name", "feature_id"],
        help="Gene identifier type in adata.var_names (default: feature_name)",
    )
    parser.add_argument(
        "--encoding-method",
        dest="encoding_method",
        choices=["singleton", "joint"],
        default="singleton",
        help=(
            "singleton: (B,1,G) per-cell (default); joint: (1,N,G) set-level "
            "(SetNorm sees all N cells at once)."
        ),
    )
    parser.add_argument(
        "--joint-chunk",
        dest="joint_chunk",
        type=int,
        default=8192,
        help="Max cells per (1,K,G) forward when encoding_method=joint (default 8192).",
    )
    parser.add_argument(
        "--load-avg",
        dest="load_avg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load averaged encoder weights (default True; matches "
            "generate_distribution_embeddings.py). Use --no-load-avg for raw weights."
        ),
    )
    args = parser.parse_args()

    extractor = OmnicellExtractor(
        checkpoint_path=args.checkpoint_path,
        base_dir=args.base_dir,
        config=args.config,
        batch_size=args.batch_size,
        device=args.device,
        gene_types=args.gene_types,
        load_avg=args.load_avg,
        encoding_method=args.encoding_method,
        joint_chunk=args.joint_chunk,
    )
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
