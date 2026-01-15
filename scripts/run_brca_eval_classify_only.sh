#!/bin/bash
# Run only evaluation and classification on pre-computed embeddings
# This script assumes embeddings have already been generated and saved in data.h5ad files
# 
# Usage:
#   cd /lotterlab/users/riccardo/ML_BIO/SCFM_meta
#   bash /lotterlab/users/riccardo/ML_BIO/scFM_repos/scFM_eval/run/run_brca_eval_classify_only.sh

# Get script directory and change to scFM_eval root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCFM_EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCFM_EVAL_DIR}" || exit 1

echo "=========================================="
echo "Running Evaluation and Classification Only"
echo "Using pre-computed embeddings"
echo "=========================================="
echo "Running from: $(pwd)"
echo ""

# Install leidenalg if not already installed
echo "Checking for leidenalg..."
uv run python -c "import leidenalg" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing leidenalg..."
    uv add leidenalg
fi
echo ""

#--------------------------------Pre vs Post (all cells)--------------------------------------------
#echo "=========================================="
#echo "Task: Pre vs Post (all cell types) - Evaluation & Classification"
#echo "=========================================="
#uv run python run/run_eval_classify_from_cache.py brca_full/pre_post/scimilarity.yaml
#uv run python run/run_eval_classify_from_cache.py brca_full/pre_post/scconcept.yaml
#echo ""

#--------------------------------subtype (ER+ vs TNBC, Cancer cells only)---------------------------------------------
echo "=========================================="
echo "Task: Subtype (ER+ vs TNBC) - Evaluation & Classification"
echo "=========================================="
uv run python run/run_eval_classify_from_cache.py brca_full/subtype/scimilarity.yaml
uv run python run/run_eval_classify_from_cache.py brca_full/subtype/scconcept.yaml
echo ""

#--------------------------------Chemo (treatment_naive vs neoadjuvant_chemo, Cancer cells only, Pre only)---------------------------------------------
echo "=========================================="
echo "Task: Chemo (treatment_naive vs neoadjuvant_chemo) - Evaluation & Classification"
echo "=========================================="
uv run python run/run_eval_classify_from_cache.py brca_full/chemo/scimilarity.yaml
uv run python run/run_eval_classify_from_cache.py brca_full/chemo/scconcept.yaml
echo ""

#-------------------------------- cell types (all cells) ---------------------------------------------
echo "=========================================="
echo "Task: Cell Type Classification (all cells) - Evaluation & Classification"
echo "=========================================="
uv run python run/run_eval_classify_from_cache.py brca_full/cell_type/scimilarity.yaml
uv run python run/run_eval_classify_from_cache.py brca_full/cell_type/scconcept.yaml
echo ""

echo "=========================================="
echo "All evaluation and classification tasks completed!"
echo "=========================================="

