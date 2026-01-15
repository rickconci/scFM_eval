#!/bin/bash
# =============================================================================
# Run LaplacianAE embeddings for Bassez (brca_full) dataset
# =============================================================================
# 
# Workflow:
#   1. Prepare dataset (combine cohorts)
#   2. Extract embeddings using LaplacianAE model
#   3. Run classification tasks using extracted embeddings
#
# Usage:
#   cd /lotterlab/users/riccardo/ML_BIO/SCFM_meta
#   CUDA_VISIBLE_DEVICES=0 bash /lotterlab/users/riccardo/ML_BIO/scFM_repos/scFM_eval/run/run_brca_laplacian_ae.sh
# =============================================================================

# Get script directory and change to scFM_eval root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCFM_EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCFM_EVAL_DIR}" || exit 1

# Set project directory for uv (where pyproject.toml is located)
UV_PROJECT_DIR="/lotterlab/users/riccardo/ML_BIO/SCFM_meta"

echo "=========================================="
echo "Running Bassez (brca_full) Evaluation"
echo "Using LaplacianAE model"
echo "=========================================="
echo "Running from: $(pwd)"
echo "UV project directory: ${UV_PROJECT_DIR}"
echo ""

# =============================================================================
# Step 1: Prepare combined dataset
# =============================================================================
echo "=========================================="
echo "Step 1: Preparing combined dataset"
echo "=========================================="
PREP_SCRIPT="/lotterlab/datasets/VCC/DATASETS/SCFM_datasets/Bassez_2021_BreastImmuno/prepare_data_for_scfm_eval.py"
OUTPUT_FILE="/lotterlab/datasets/VCC/DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad"

(cd "${UV_PROJECT_DIR}" && uv run python "${PREP_SCRIPT}")

if [ -f "${OUTPUT_FILE}" ]; then
    size_mb=$(du -m "${OUTPUT_FILE}" 2>/dev/null | cut -f1)
    echo "✓ Dataset file ready! (${size_mb} MB)"
else
    echo "ERROR: Dataset file not found. Exiting."
    exit 1
fi
echo ""

# =============================================================================
# Step 2: Run classification tasks
# =============================================================================
echo "=========================================="
echo "Step 2: Running classification tasks with LaplacianAE embeddings"
echo "=========================================="
echo ""

#-------------------------------- pre_post ---------------------------------------------
echo "----------------------------------------"
echo "Task: Pre vs Post Treatment"
echo "----------------------------------------"
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/pre_post/laplacian_ae.yaml)
echo ""

#-------------------------------- cell types ---------------------------------------------
echo "----------------------------------------"
echo "Task: Cell Type Classification"
echo "----------------------------------------"
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/cell_type/laplacian_ae.yaml)
echo ""

#-------------------------------- all_cells (pre_post) ---------------------------------------------
echo "----------------------------------------"
echo "Task: All Cells - Pre vs Post Treatment"
echo "----------------------------------------"
(cd "${UV_PROJECT_DIR}" && uv run python "${SCFM_EVAL_DIR}/run/run_exp.py" brca_full/all_cells/laplacian_ae.yaml)
echo ""

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
