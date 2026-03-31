"""Run batch/bio/annotation experiments across Omnicell checkpoints.

This script sweeps a list of Omnicell checkpoint paths and runs selected methods
on the batch_bio_abugoot_no_tabula task.

Output layout (default):
  <output_root>/harmony/
  <output_root>/pca_qc/
  <output_root>/omnicell_<checkpoint_stem>/
so checkpoint-specific Omnicell runs never overwrite each other, while harmony
and pca_qc remain shared baselines.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import signal
from pathlib import Path
from typing import Sequence


DEFAULT_CHECKPOINT_ROOT = Path("/orcd/data/omarabu/001/njwfish")
DEFAULT_CHECKPOINTS = [
    "cell-types/cell_types/outputs/obs/517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_3.pt",
    "cell-types/cell_types/outputs/obs/517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_4.pt",
    "cell-types/cell_types/outputs/obs/517d544b28d4a8b92f641f7627edaeff/checkpoint_epoch_5.pt",
    "cell-types/cell_types/outputs/obs/5ce1b9325f67e5aaebca59c70cc48b89/checkpoint_epoch_3.pt",
    "cell-types/cell_types/outputs/obs/5ce1b9325f67e5aaebca59c70cc48b89/checkpoint_epoch_2.pt",
]


ACTIVE_CHILD_PGIDS: set[int] = set()


def _kill_active_children() -> None:
    """Kill all active subprocess groups started by this script."""
    for pgid in list(ACTIVE_CHILD_PGIDS):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to SIGTERM pgid {pgid}: {exc}", file=sys.stderr)
    # Small grace period then hard kill any survivors
    for pgid in list(ACTIVE_CHILD_PGIDS):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to SIGKILL pgid {pgid}: {exc}", file=sys.stderr)
    ACTIVE_CHILD_PGIDS.clear()


def _signal_handler(signum: int, _frame: object) -> None:
    """Handle Ctrl+C / termination by cleaning up child process trees."""
    print(f"\nReceived signal {signum}. Stopping active parallel runs...", file=sys.stderr)
    _kill_active_children()
    raise SystemExit(130)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Omnicell checkpoints over batch_bio_abugoot_no_tabula."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional list of Omnicell checkpoint .pt paths to sweep.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Base directory used to resolve relative default checkpoint paths.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/orcd/data/omarabu/001/rconci/scFM_eval"),
        help="Path to scFM_eval repository root.",
    )
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=Path("/orcd/data/omarabu/001/rconci/scFM_eval/yaml"),
        help="YAML root passed to parallel_experiments.main --yaml-dir.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="batch_bio_abugoot_no_tabula",
        help="Task folder under yaml/ to run.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="omnicell,harmony,pca_qc",
        help="Comma-separated methods for --method filter.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/orcd/data/omarabu/001/Omnicell_datasets/bio_batch_eval_results/omnicell_checkpoint_results"),
        help="Root output directory. Each checkpoint writes to a child directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for parallel_experiments.main.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Optional GPU list, e.g. '0,1'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to rerun experiments even if results exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to the parallel runner.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help=(
            "If a checkpoint batch (or harmony/pca_qc run) exits non-zero, still run the "
            "remaining checkpoints and baseline methods. Exit code is non-zero if any "
            "phase failed. Default: stop the whole sweep on first failure (parallel "
            "runner returns 1 when any YAML in that batch fails)."
        ),
    )
    return parser.parse_args()


def _validate_checkpoints(checkpoints: Sequence[str], checkpoint_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for ckpt in checkpoints:
        p = Path(ckpt).expanduser()
        if not p.is_absolute():
            p = checkpoint_root / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        resolved.append(p)
    return resolved


def _build_cmd(
    task: str,
    yaml_dir: Path,
    methods: str,
    output_dir: Path,
    workers: int,
    gpus: str | None,
    force: bool,
    dry_run: bool,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "parallel_experiments.main",
        "--tasks",
        task,
        "--yaml-dir",
        str(yaml_dir),
        "--method",
        methods,
        "--output-dir",
        str(output_dir),
        "-w",
        str(workers),
    ]
    if gpus:
        cmd.extend(["--gpus", gpus])
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_one(
    *,
    repo_dir: Path,
    yaml_dir: Path,
    task: str,
    methods: str,
    output_dir: Path,
    workers: int,
    gpus: str | None,
    force: bool,
    dry_run: bool,
    env: dict[str, str],
    label: str,
) -> int:
    """Run one parallel_experiments invocation with logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(
        task=task,
        yaml_dir=yaml_dir,
        methods=methods,
        output_dir=output_dir,
        workers=workers,
        gpus=gpus,
        force=force,
        dry_run=dry_run,
    )
    print("\n" + "=" * 80)
    print(label)
    print(f"Output dir: {output_dir}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        env=env,
        text=True,
        start_new_session=True,  # child gets its own process group
    )
    try:
        ACTIVE_CHILD_PGIDS.add(proc.pid)
        return proc.wait()
    except KeyboardInterrupt:
        print("\nInterrupted. Killing active parallel run tree...", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        return 130
    finally:
        ACTIVE_CHILD_PGIDS.discard(proc.pid)


def _checkpoint_namespace(ckpt: Path) -> str:
    """Map checkpoint file name to output model namespace."""
    stem = ckpt.stem  # e.g. checkpoint_epoch_2
    prefix = "checkpoint_epoch_"
    if stem.startswith(prefix):
        epoch = stem[len(prefix):]
        if epoch.isdigit():
            return f"omnicell_checkpoint_{epoch}"
    return f"omnicell_{stem}"


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    args = _parse_args()

    repo_dir = args.repo_dir.resolve()
    yaml_dir = args.yaml_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_list = args.checkpoints if args.checkpoints else DEFAULT_CHECKPOINTS
    checkpoints = _validate_checkpoints(checkpoint_list, args.checkpoint_root.resolve())

    print(f"Running {len(checkpoints)} checkpoint(s)")
    print(f"Task: {args.task}")
    print(f"Requested methods: {args.methods}")
    print(f"YAML root: {yaml_dir}")
    print(f"Output root: {output_root}")
    print("Output layout: <output_root>/<task>/<namespace>/<batch_type>/<dataset>/...")

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    shared_methods = [m for m in requested_methods if m in {"harmony", "pca_qc"}]
    run_omnicell = "omnicell" in requested_methods
    any_failure = False

   

    # Run Omnicell once per checkpoint first
    if run_omnicell:
        for i, ckpt in enumerate(checkpoints, start=1):
            namespace = _checkpoint_namespace(ckpt)
            env = os.environ.copy()
            env["OMNICELL_CHECKPOINT_PATH"] = str(ckpt)
            env["RESULTS_METHOD_NAMESPACE"] = namespace
            rc = _run_one(
                repo_dir=repo_dir,
                yaml_dir=yaml_dir,
                task=args.task,
                methods="omnicell",
                output_dir=output_root,
                workers=args.workers,
                gpus=args.gpus,
                force=args.force,
                dry_run=args.dry_run,
                env=env,
                label=f"[{i}/{len(checkpoints)}] Omnicell checkpoint: {ckpt} (namespace={namespace})",
            )
            if rc != 0:
                print(f"Checkpoint run failed (exit={rc}): {ckpt}", file=sys.stderr)
                any_failure = True
                if not args.continue_on_failure:
                    return rc

    # Then run shared baselines once (not checkpoint-dependent)
    for method in shared_methods:
        namespace = method.lower()
        env = os.environ.copy()
        env["RESULTS_METHOD_NAMESPACE"] = namespace
        rc = _run_one(
            repo_dir=repo_dir,
            yaml_dir=yaml_dir,
            task=args.task,
            methods=method,
            output_dir=output_root,
            workers=args.workers,
            gpus=args.gpus,
            force=args.force,
            dry_run=args.dry_run,
            env={**os.environ.copy(), "RESULTS_METHOD_NAMESPACE": method},
            label=f"[shared] Method: {method}",
        )
        if rc != 0:
            print(
                f"Shared baseline run failed (exit={rc}) for method: {method}",
                file=sys.stderr,
            )
            any_failure = True
            if not args.continue_on_failure:
                return rc

    if any_failure:
        print(
            "\nSweep finished with at least one failed phase (see errors above). "
            "Tip: pass --continue-on-failure to run later checkpoints/baselines when "
            "an earlier batch has partial failures.",
            file=sys.stderr,
        )
        return 1

    print("\nAll checkpoint runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
