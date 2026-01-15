#!/usr/bin/env python3
"""
Parallel experiment runner for scFM_eval.

Runs multiple YAML configs in parallel across N workers.

Usage:
    python run_experiments_parallel.py [options]

Examples:
    # Run all experiments with 8 workers
    python run_experiments_parallel.py -w 8

    # Run only batch_denoising task
    python run_experiments_parallel.py -t batch_denoising

    # Run only scimilarity method on gtex_v9
    python run_experiments_parallel.py -m scimilarity -d gtex_v9

    # Dry run to see what would be executed
    python run_experiments_parallel.py --dry-run
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from setup_path import OUTPUT_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    yaml_path: Path
    success: bool
    skipped: bool = False
    dry_run: bool = False
    error_message: str | None = None
    duration_seconds: float = 0.0


def find_yaml_files(
    yaml_dir: Path,
    task_filter: str | None = None,
    method_filter: str | None = None,
    dataset_filter: str | None = None,
) -> list[Path]:
    """
    Find all YAML config files matching the filters.

    Args:
        yaml_dir: Base directory containing YAML files.
        task_filter: Only include files from this task (e.g., 'batch_denoising').
        method_filter: Only include files from this method (e.g., 'scimilarity').
        dataset_filter: Only include files with this dataset name.

    Returns:
        List of paths to matching YAML files.
    """
    yaml_files = []

    # Find files in scimilarity or scconcept subdirectories
    for pattern in ["*/scimilarity/*.yaml", "*/scconcept/*.yaml"]:
        for yaml_path in yaml_dir.glob(pattern):
            rel_path = yaml_path.relative_to(yaml_dir)
            parts = rel_path.parts

            if len(parts) < 3:
                continue

            task = parts[0]
            method = parts[1]
            dataset = yaml_path.stem

            # Apply filters
            if task_filter and task != task_filter:
                continue
            if method_filter and method != method_filter:
                continue
            if dataset_filter and dataset != dataset_filter:
                continue

            yaml_files.append(yaml_path)

    return sorted(yaml_files)


def check_existing_results(yaml_path: Path, yaml_dir: Path) -> bool:
    """
    Check if experiment results already exist.

    Args:
        yaml_path: Path to the YAML config file.
        yaml_dir: Base YAML directory.

    Returns:
        True if results exist, False otherwise.
    """
    rel_path = yaml_path.relative_to(yaml_dir)
    parts = rel_path.parts

    if len(parts) < 3:
        return False

    task = parts[0]
    method = parts[1]
    dataset = yaml_path.stem

    output_dir = OUTPUT_PATH / task / method / dataset

    if output_dir.exists():
        # Check if any metrics CSV files exist
        metrics_files = list(output_dir.glob("**/metrics/*.csv"))
        return len(metrics_files) > 0

    return False


def run_single_experiment(
    yaml_path: Path,
    base_dir: Path,
    yaml_dir: Path,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> ExperimentResult:
    """
    Run a single experiment.

    Args:
        yaml_path: Path to the YAML config file.
        base_dir: Base directory of the project.
        yaml_dir: Base YAML directory.
        skip_existing: Skip if results already exist.
        dry_run: Don't actually run, just report what would be done.

    Returns:
        ExperimentResult with success/failure status.
    """
    rel_path = yaml_path.relative_to(yaml_dir)
    start_time = datetime.now()

    # Check if should skip
    if skip_existing and check_existing_results(yaml_path, yaml_dir):
        return ExperimentResult(
            yaml_path=yaml_path,
            success=True,
            skipped=True,
        )

    # Dry run mode
    if dry_run:
        return ExperimentResult(
            yaml_path=yaml_path,
            success=True,
            dry_run=True,
        )

    # Run the experiment
    try:
        result = subprocess.run(
            [sys.executable, "run/run_exp.py", str(rel_path)],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per experiment
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            return ExperimentResult(
                yaml_path=yaml_path,
                success=True,
                duration_seconds=duration,
            )
        else:
            return ExperimentResult(
                yaml_path=yaml_path,
                success=False,
                error_message=result.stderr[:500] if result.stderr else "Unknown error",
                duration_seconds=duration,
            )

    except subprocess.TimeoutExpired:
        return ExperimentResult(
            yaml_path=yaml_path,
            success=False,
            error_message="Timeout after 1 hour",
            duration_seconds=3600,
        )
    except Exception as e:
        return ExperimentResult(
            yaml_path=yaml_path,
            success=False,
            error_message=str(e),
        )


def run_experiment_wrapper(args: tuple) -> ExperimentResult:
    """Wrapper for ProcessPoolExecutor (unpacks tuple args)."""
    return run_single_experiment(*args)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel experiment runner for scFM_eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        help="Run only specific task (e.g., batch_denoising)",
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default=None,
        help="Run only specific method (e.g., scimilarity)",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Run only specific dataset (e.g., gtex_v9)",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Run only specific YAML file (relative to yaml/)",
    )
    parser.add_argument(
        "-s", "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.resolve()
    yaml_dir = base_dir / "yaml"

    # Find YAML files
    if args.file:
        yaml_path = yaml_dir / args.file
        if not yaml_path.exists():
            logger.error(f"File not found: {yaml_path}")
            return 1
        yaml_files = [yaml_path]
    else:
        yaml_files = find_yaml_files(
            yaml_dir,
            task_filter=args.task,
            method_filter=args.method,
            dataset_filter=args.dataset,
        )

    if not yaml_files:
        logger.error("No YAML files found matching the criteria!")
        return 1

    # Log configuration
    logger.info(f"Found {len(yaml_files)} YAML config files")
    if args.dry_run:
        logger.info("DRY RUN MODE - No experiments will be executed")
    else:
        logger.info(f"Using {args.workers} parallel workers")
    if args.skip_existing:
        logger.info("Will skip experiments that already have results")

    # Prepare arguments for parallel execution
    experiment_args = [
        (yaml_path, base_dir, yaml_dir, args.skip_existing, args.dry_run)
        for yaml_path in yaml_files
    ]

    # Run experiments in parallel
    results: list[ExperimentResult] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_experiment_wrapper, exp_args): exp_args[0]
            for exp_args in experiment_args
        }

        for future in as_completed(futures):
            yaml_path = futures[future]
            rel_path = yaml_path.relative_to(yaml_dir)

            try:
                result = future.result()
                results.append(result)

                if result.dry_run:
                    logger.info(f"[DRY RUN] Would run: {rel_path}")
                elif result.skipped:
                    logger.info(f"⏭ Skipped (exists): {rel_path}")
                elif result.success:
                    logger.info(f"✓ Success: {rel_path} ({result.duration_seconds:.1f}s)")
                else:
                    logger.error(f"✗ Failed: {rel_path} - {result.error_message}")

            except Exception as e:
                logger.error(f"✗ Exception for {rel_path}: {e}")
                results.append(ExperimentResult(
                    yaml_path=yaml_path,
                    success=False,
                    error_message=str(e),
                ))

    # Summary
    dry_run_count = sum(1 for r in results if r.dry_run)
    success_count = sum(1 for r in results if r.success and not r.skipped and not r.dry_run)
    failed_count = sum(1 for r in results if not r.success)
    skipped_count = sum(1 for r in results if r.skipped)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total:   {len(results)}")
    
    if args.dry_run:
        print(f"Would run: {dry_run_count}")
        print(f"Would skip (existing): {skipped_count}")
    else:
        print(f"Success: {success_count}")
        print(f"Failed:  {failed_count}")
        print(f"Skipped: {skipped_count}")

    if failed_count > 0:
        print("\nFailed experiments:")
        for result in results:
            if not result.success:
                rel_path = result.yaml_path.relative_to(yaml_dir)
                print(f"  - {rel_path}: {result.error_message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
