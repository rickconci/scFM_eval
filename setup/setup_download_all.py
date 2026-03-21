#!/usr/bin/env python3
"""
One-shot setup: download datasets, clone model repos, and fetch checkpoints.

Reads dataset_paths.yaml and model_paths.yaml from the setup/ directory. Requires
DATASETS_PATH and MODEL_CHECKPOINTS in environment (e.g. from .env loaded by the caller).

Usage:
  python setup/setup_download_all.py
  python setup/setup_download_all.py --datasets-only
  python setup/setup_download_all.py --repos-only
  python setup/setup_download_all.py --checkpoints-only
  python setup/setup_download_all.py -v   # debug (URLs, paths)
  python setup/setup_download_all.py -q   # warnings/errors only
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import dotenv
from huggingface_hub import snapshot_download, hf_hub_download
import gdown

SETUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SETUP_DIR.parent
env_path = REPO_ROOT / ".env"
dotenv.load_dotenv(env_path)

logger = logging.getLogger(__name__)


def configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Configure root logging for this script.

    Args:
        verbose: If True, emit DEBUG messages.
        quiet: If True, only WARNING and above. Overrides verbose when both are set.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; require PyYAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_url(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with streaming I/O (safe for multi‑GiB files).

    ``urllib``'s ``read()`` loads the whole body into RAM; Zenodo checkpoints can be
    tens of GiB, so we write in chunks.

    Args:
        url: HTTP(S) URL to fetch.
        dest: Destination path (parent dirs are created).

    Raises:
        urllib.error.URLError: On network or HTTP errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("GET %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "scFM_eval-setup/1.0"})
    chunk_size = 8 * 1024 * 1024
    log_interval = 512 * 1024 * 1024
    with urllib.request.urlopen(req) as resp:
        cl = resp.headers.get("Content-Length", "")
        total: int | None = int(cl) if cl.isdigit() else None
        if total is not None:
            logger.info(
                "Streaming download (~%.2f GiB) -> %s",
                total / (1024**3),
                dest,
            )
        written = 0
        last_log_at = 0
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if written - last_log_at >= log_interval:
                    if total is not None:
                        logger.info(
                            "Progress %s: %.1f / %.1f GiB (%.0f%%)",
                            dest.name,
                            written / (1024**3),
                            total / (1024**3),
                            100.0 * written / total,
                        )
                    else:
                        logger.info(
                            "Progress %s: %.2f GiB so far (total size unknown)",
                            dest.name,
                            written / (1024**3),
                        )
                    last_log_at = written
    logger.info("Finished download %s (%d bytes)", dest, written)


def do_datasets(datasets_path: Path, manifest: dict[str, Any]) -> None:
    """Download all datasets from manifest to datasets_path."""
    entries = manifest.get("datasets", [])
    logger.info("Downloading %d dataset(s) to %s", len(entries), datasets_path)
    for item in entries:
        url = item["url"]
        rel = item["path"]
        dest = datasets_path / rel
        name = item.get("name", item.get("id", rel))
        if dest.exists():
            logger.info("Dataset %r already present at %s, skip", name, dest)
            continue
        logger.info("Downloading dataset %r -> %s", name, dest)
        download_url(url, dest)
    logger.info("Datasets phase finished.")


def do_repos(repos_dir: Path, manifest: dict[str, Any]) -> None:
    """Clone all repos from manifest into repos_dir."""
    entries = manifest.get("repos", [])
    logger.info("Cloning %d repo(s) under %s", len(entries), repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    for item in entries:
        name = item["name"]
        url = item["url"]
        clone_dir = repos_dir / name
        if (clone_dir / ".git").exists():
            logger.info("Repo %r already cloned at %s, skip", name, clone_dir)
            continue
        logger.info("Shallow clone %r from %s -> %s", name, url, clone_dir)
        shallow = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(clone_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if shallow.returncode != 0:
            logger.warning(
                "Shallow clone failed for %r (exit %d): %s",
                name,
                shallow.returncode,
                (shallow.stderr or shallow.stdout or "").strip() or "(no output)",
            )
        if not (clone_dir / ".git").exists():
            logger.info("Full clone %r -> %s", name, clone_dir)
            full = subprocess.run(
                ["git", "clone", url, str(clone_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            if full.returncode != 0:
                logger.error(
                    "git clone failed for %r (exit %d): %s",
                    name,
                    full.returncode,
                    (full.stderr or full.stdout or "").strip() or "(no output)",
                )
                full.check_returncode()
    logger.info("Repos phase finished.")


def do_checkpoints(checkpoints_base: Path, manifest: dict[str, Any]) -> None:
    """Download checkpoints listed in the manifest under checkpoints_base."""
    entries = manifest.get("checkpoints", [])
    logger.info("Processing %d checkpoint(s) under %s", len(entries), checkpoints_base)
    checkpoints_base.mkdir(parents=True, exist_ok=True)

    for item in entries:
        ckpt_type = item.get("type", "")
        name = item.get("name", "unknown")
        local_dir = checkpoints_base / item.get("local_dir", name)
        logger.debug("Checkpoint %r type=%r local_dir=%s", name, ckpt_type, local_dir)

        # 1. HuggingFace Full Repository
        if ckpt_type == "huggingface":
            if local_dir.exists() and any(local_dir.iterdir()):
                logger.info("HF repo %r already present at %s, skip", name, local_dir)
                continue
            repo_id = item["repo_id"]
            logger.info("Downloading HF snapshot %r (repo_id=%s) -> %s", name, repo_id, local_dir)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
            )
            logger.info("Finished HF snapshot %r", name)

        # 2. HuggingFace Single File
        elif ckpt_type == "huggingface_file":
            target_file = local_dir / item["file"]
            if target_file.exists():
                logger.info("HF file %r already present at %s, skip", name, target_file)
                continue
            repo_id = item["repo_id"]
            filename = item["file"]
            logger.info(
                "Downloading HF file %r (repo_id=%s, file=%s) -> %s",
                name,
                repo_id,
                filename,
                local_dir,
            )
            local_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            logger.info("Finished HF file %r", name)

        # 3. Google Drive Folders/Files
        elif ckpt_type == "gdrive":
            if local_dir.exists() and any(local_dir.iterdir()):
                logger.info("GDrive %r already present at %s, skip", name, local_dir)
                continue
            url = item["url"]
            logger.info("Downloading GDrive %r -> %s", name, local_dir)
            if "drive.google.com/drive/folders" in url:
                gdown.download_folder(url=url, output=str(local_dir), quiet=False)
            else:
                gdown.download(url=url, output=str(local_dir / "checkpoint.ckpt"), quiet=False)
            logger.info("Finished GDrive %r", name)

        # 4. Standard Direct URL
        elif ckpt_type == "url":
            local_dir.mkdir(parents=True, exist_ok=True)
            url = item["url"]
            filename = url.split("/")[-1].split("?")[0] or "downloaded"
            dest_file = local_dir / filename
            if dest_file.exists():
                logger.info("URL checkpoint %r already present at %s, skip", name, dest_file)
                continue
            logger.info("Downloading URL checkpoint %r -> %s", name, dest_file)
            download_url(url, dest_file)
            is_tar_gz = dest_file.suffix in (".tgz",) or dest_file.name.endswith(".tar.gz")
            is_gz_archive = dest_file.suffix == ".gz" and is_tar_gz
            if item.get("extract") and (is_tar_gz or dest_file.suffix == ".tgz"):
                logger.info("Extracting %s into %s (this may take a while for large archives)", dest_file, local_dir)
                with tarfile.open(dest_file, "r:gz") as tf:
                    tf.extractall(local_dir)
                dest_file.unlink()
                logger.info("Removed archive after extract: %s", dest_file)
            elif item.get("extract") and not (is_tar_gz or dest_file.suffix == ".tgz"):
                logger.warning(
                    "extract=true but %s is not a .tar.gz / .tgz; leaving archive on disk",
                    dest_file,
                )
        elif ckpt_type == "instructions_only":
            msg = item.get("message", "No automated download; see model docs.")
            logger.info("Checkpoint %r: %s", name, msg)
        else:
            logger.warning("Unknown checkpoint type %r for %r, skipping", ckpt_type, name)

    logger.info("Checkpoints phase finished.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets, clone repos, fetch checkpoints")
    parser.add_argument("--datasets-only", action="store_true", help="Only download datasets")
    parser.add_argument("--repos-only", action="store_true", help="Only clone repos")
    parser.add_argument("--checkpoints-only", action="store_true", help="Only fetch checkpoints")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug details (URLs, paths, HF repo ids)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, quiet=args.quiet)
    logger.debug("Loaded .env from %s (if present)", env_path)

    do_datasets_run = args.datasets_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )
    do_repos_run = args.repos_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )
    do_checkpoints_run = args.checkpoints_only or (
        not args.datasets_only and not args.repos_only and not args.checkpoints_only
    )

    datasets_path = os.getenv("DATASETS_PATH")
    model_checkpoints = os.getenv("MODEL_CHECKPOINTS")
    if not datasets_path and do_datasets_run:
        raise SystemExit("Set DATASETS_PATH in .env (root for downloaded .h5ad files)")
    if not model_checkpoints and (do_repos_run or do_checkpoints_run):
        raise SystemExit("Set MODEL_CHECKPOINTS in .env (root for model checkpoints)")

    datasets_path = Path(datasets_path) if datasets_path else None
    model_checkpoints = Path(model_checkpoints) if model_checkpoints else None
    repos_dir = (
        Path(os.environ.get("REPOS_DIR", str(model_checkpoints.parent / "repos")))
        if model_checkpoints
        else Path(os.environ.get("REPOS_DIR", "."))
    )
    logger.info(
        "Paths: datasets_path=%s model_checkpoints=%s repos_dir=%s",
        datasets_path,
        model_checkpoints,
        repos_dir,
    )

    dataset_manifest_path = SETUP_DIR / "dataset_paths.yaml"
    model_manifest_path = SETUP_DIR / "model_paths.yaml"

    if do_datasets_run and datasets_path:
        if dataset_manifest_path.exists():
            do_datasets(datasets_path, load_yaml(dataset_manifest_path))
        else:
            logger.warning("Dataset manifest missing, skipping datasets: %s", dataset_manifest_path)

    if do_repos_run and model_checkpoints is not None:
        if model_manifest_path.exists():
            do_repos(repos_dir, load_yaml(model_manifest_path))
        else:
            logger.warning("Model manifest missing, skipping repos: %s", model_manifest_path)

    if do_checkpoints_run and model_checkpoints:
        if model_manifest_path.exists():
            do_checkpoints(model_checkpoints, load_yaml(model_manifest_path))
        else:
            logger.warning(
                "Model manifest missing, skipping checkpoints: %s",
                model_manifest_path,
            )

    logger.info("Setup finished. Set DATASETS_PATH and MODEL_CHECKPOINTS in .env and run your eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
