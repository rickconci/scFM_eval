#!/usr/bin/env python
"""Smoke test: load only the Omnicell encoder from a checkpoint and run one forward pass.

``run_omnicell_embed.py`` already does the full pipeline (OmnicellBase → H5AD → ``map_genes`` →
encoder). This script is smaller: it uses ``generative.utils.loading.load_encoder`` so the
**generator is not loaded**, and uses random input of the correct width
(``len(gene_manager["shared_genes"])``) to verify weights and CUDA/CPU execution.

Requires the same environment as cell-types Omnicell (``OMNICELL_DATA_DIR`` for
``gene_manager.pkl``, Hydra configs under ``<cell_types>/generative/config``).

Example::

    export OMNICELL_DATA_DIR=/path/to/dir/with/protocol_embeddings
    PYTHONPATH=/path/to/cell-types:/path/to/cell-types/cell_types \\
      python test_encoder_smoke.py \\
        --checkpoint_path /path/to/checkpoint.pt \\
        --base_dir /path/to/cell-types/cell_types

Or from ``scFM_eval`` with a ``ml`` (or cell-types) env::

    python extractors/omnicell/test_encoder_smoke.py \\
      --checkpoint_path \"$OMNICELL_CHECKPOINT_PATH\" \\
      --base_dir \"$OMNICELL_BASE_DIR\"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Tuple

import numpy as np
import torch

import dotenv
env_path = '/orcd/data/omarabu/001/rconci/scFM_eval/.env'
dotenv.load_dotenv(env_path)

def _resolve_package_dir(base_dir: Path) -> Tuple[Path, Path]:
    """Return ``(repo_root, package_dir)`` for cell-types layout."""
    base_path = base_dir.resolve()
    if (base_path / "cell_types").is_dir():
        return base_path, base_path / "cell_types"
    if base_path.name == "cell_types":
        return base_path.parent, base_path
    return base_path.parent, base_path


def _setup_sys_path(package_dir: Path) -> Path:
    """Prepend repo and package paths so ``generative`` and ``cell_types`` imports work."""
    repo_root, pkg = _resolve_package_dir(package_dir)
    path_add = [str(repo_root), str(pkg)]
    bulk = pkg / "bulk_prediction"
    if bulk.is_dir():
        path_add.append(str(bulk))
    gen_root = pkg / "generative"
    if gen_root.is_dir():
        path_add.append(str(gen_root))
    for p in path_add:
        if p not in sys.path:
            sys.path.insert(0, p)
    return pkg


def _load_constants_from_package_dir(package_dir: Path) -> ModuleType:
    """Load ``constants.py`` from the inner cell-types tree by path (avoids ``import constants`` conflicts).

    ``cell_types/tasks/core/base.py`` uses ``from constants import ...`` because that package dir is on
    ``sys.path``. Loading by file is deterministic and works even if another ``constants`` is installed.
    """
    path = package_dir.resolve() / "constants.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected cell-types package at {package_dir} (missing {path}). "
            "Pass --base_dir /path/to/cell-types/cell_types. "
            "If you use .env, run: set -a && source .env && set +a "
            "or: export OMNICELL_BASE_DIR=/path/to/cell-types/cell_types"
        ) from None
    spec = importlib.util.spec_from_file_location("omnicell_constants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}") from None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load Omnicell encoder only and run a tiny forward pass (smoke test)."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help=(
            "Checkpoint with encoder_state_dict and/or avg_encoder_state_dict "
            "(default: load averaged weights, same as generate_distribution_embeddings.py)"
        ),
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="cell_types package directory (cell-types/cell_types)",
    )
    parser.add_argument("--config", default="obs", help="Hydra config name (default: obs)")
    parser.add_argument(
        "--load-avg",
        dest="load_avg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load averaged encoder weights (avg_encoder_state_dict); default True, "
            "matching generate_distribution_embeddings.py. Use --no-load-avg for "
            "raw encoder_state_dict."
        ),
    )
    parser.add_argument(
        "--batch_cells",
        type=int,
        default=4,
        help="Number of fake cells in the forward pass (default: 4)",
    )
    parser.add_argument(
        "--omnicell_data_dir",
        default=None,
        help="If set, exported as OMNICELL_DATA_DIR before imports (gene_manager path)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        print(f"error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    if not str(args.base_dir).strip():
        print(
            "error: --base_dir is empty. Set OMNICELL_BASE_DIR or pass the path to "
            "cell-types/cell_types (e.g. export from .env before running).",
            file=sys.stderr,
        )
        return 2

    if args.omnicell_data_dir:
        os.environ["OMNICELL_DATA_DIR"] = args.omnicell_data_dir.rstrip("/") + "/"

    base_path = Path(args.base_dir).resolve()
    _, package_dir = _resolve_package_dir(base_path)
    # Same order as run_omnicell_embed: path first, then constants (reads OMNICELL_DATA_DIR at import).
    _setup_sys_path(base_path)
    constants_mod = _load_constants_from_package_dir(package_dir)
    gene_manager_path = constants_mod.GENE_MANAGER_PATH

    t0 = time.perf_counter()
    with open(gene_manager_path, "rb") as f:
        gene_manager = pickle.load(f)
    n_shared = len(gene_manager["shared_genes"])
    print(f"[smoke] gene_manager: {gene_manager_path} ({n_shared} shared genes)", flush=True)

    from generative.utils.loading import load_encoder  # noqa: PLC0415

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[smoke] loading encoder config={args.config!r} device={device} ...", flush=True)
    encoder = load_encoder(
        config=args.config,
        base_dir=str(package_dir),
        gene_manager=gene_manager,
        checkpoint=str(checkpoint_path),
        load_avg=args.load_avg,
        device=device,
    )
    print(f"[smoke] load_encoder done in {time.perf_counter() - t0:.2f}s", flush=True)

    b = max(1, args.batch_cells)
    x = torch.randn(b, 1, n_shared, device=device, dtype=torch.float32)
    with torch.no_grad():
        lat, cell_emb, _ = encoder(x, return_cell_embeddings=True)

    lat_np = lat.cpu().numpy()
    cell_np = cell_emb.cpu().numpy()
    print(f"[smoke] forward ok: lat {lat_np.shape} cell_emb {cell_np.shape}", flush=True)
    print(
        f"[smoke] lat mean={float(lat_np.mean()):.4f} std={float(lat_np.std()):.4f} "
        f"finite={bool(np.isfinite(lat_np).all())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
