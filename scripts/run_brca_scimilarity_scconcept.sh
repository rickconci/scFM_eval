#!/bin/bash
# Run only scSimilarity and scConcept for Bassez (brca_full) dataset
# Uses combined Cohort 1 + Cohort 2 dataset
# Cohort 1 = treatment_naive, Cohort 2 = neoadjuvant_chemo
# Skips outcome task (no proper outcome data available)
# 
# Usage:
#   cd /lotterlab/users/riccardo/ML_BIO/SCFM_meta
#   bash /lotterlab/users/riccardo/ML_BIO/scFM_repos/scFM_eval/run/run_brca_scimilarity_scconcept.sh

# Get script directory and change to scFM_eval root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCFM_EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCFM_EVAL_DIR}" || exit 1

# Set project directory for uv (where pyproject.toml is located)
UV_PROJECT_DIR="/lotterlab/users/riccardo/ML_BIO/SCFM_meta"

echo "=========================================="
echo "Running Bassez (brca_full) Evaluation"
echo "Using combined Cohort 1 + Cohort 2 dataset"
echo "=========================================="
echo "Running from: $(pwd)"
echo "UV project directory: ${UV_PROJECT_DIR}"
echo ""

#--------------------------------Prepare dataset (combines cohorts, saves in background)--------------------------------------------
echo "=========================================="
echo "Step 1: Preparing combined dataset"
echo "=========================================="
PREP_SCRIPT="/lotterlab/datasets/VCC/DATASETS/SCFM_datasets/Bassez_2021_BreastImmuno/prepare_data_for_scfm_eval.py"
OUTPUT_FILE="/lotterlab/datasets/VCC/DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad"

# Run preparation script (saves in background thread)
(cd "${UV_PROJECT_DIR}" && uv run python "${PREP_SCRIPT}")

# Check if file already exists (skip wait if already processed)
if [ -f "${OUTPUT_FILE}" ]; then
    size_mb=$(du -m "${OUTPUT_FILE}" 2>/dev/null | cut -f1)
    echo "✓ Dataset file already exists! (${size_mb} MB)"
    echo "  Skipping preparation step."
    echo ""
else
    # Wait for file to exist
    echo "Waiting for dataset file to be ready..."
    MAX_WAIT=600  # Maximum wait time in seconds (10 minutes)
    WAIT_INTERVAL=2  # Check every 2 seconds
    elapsed=0

    while [ ! -f "${OUTPUT_FILE}" ] && [ $elapsed -lt $MAX_WAIT ]; do
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
        if [ $((elapsed % 20)) -eq 0 ]; then
            echo "  Still waiting for file to appear... (${elapsed}s elapsed)"
        fi
    done

    if [ ! -f "${OUTPUT_FILE}" ]; then
        echo "ERROR: Dataset file not found after ${MAX_WAIT}s. Exiting."
        exit 1
    else
        size_mb=$(du -m "${OUTPUT_FILE}" 2>/dev/null | cut -f1)
        echo "✓ Dataset file is ready! (${size_mb} MB, saved in ${elapsed}s)"
    fi
    echo ""
fi

#--------------------------------Pre vs Post (all cells)--------------------------------------------
echo "=========================================="
echo "Task: Pre vs Post (all cell types)"
echo "=========================================="
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/pre_post/scimilarity.yaml)
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/pre_post/scconcept.yaml)
echo ""

#--------------------------------subtype (ER+ vs TNBC, Cancer cells only)---------------------------------------------
echo "=========================================="
echo "Task: Subtype (ER+ vs TNBC)"
echo "=========================================="
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/subtype/scimilarity.yaml)
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/subtype/scconcept.yaml)
echo ""

#--------------------------------SKIP outcome - no proper clinical outcome data available---------------------------------------------
# echo "=========================================="
# echo "Task: Outcome (SKIPPED - no proper clinical outcome data)"
# echo "=========================================="
# uv run python run/run_exp.py brca_full/outcome/scimilarity.yaml
# uv run python run/run_exp.py brca_full/outcome/scconcept.yaml

#--------------------------------Chemo (treatment_naive vs neoadjuvant_chemo, Cancer cells only, Pre only)---------------------------------------------
echo "=========================================="
echo "Task: Chemo (treatment_naive vs neoadjuvant_chemo)"
echo "=========================================="
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/chemo/scimilarity.yaml)
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/chemo/scconcept.yaml)
echo ""

#-------------------------------- cell types (all cells) ---------------------------------------------
echo "=========================================="
echo "Task: Cell Type Classification (all cells)"
echo "=========================================="
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/cell_type/scimilarity.yaml)
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/cell_type/scconcept.yaml)
echo ""

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="

