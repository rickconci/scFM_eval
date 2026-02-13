# scFM_eval

**scFM_eval** is an evaluation framework for single-cell foundation models. It runs a standard pipeline so you can compare how well different embedding models perform on the same datasets and tasks.


**Core Functionality**

1. **Loads a dataset** (e.g. an `.h5ad` file) and optional QC/preprocessing (normalization, HVG, train/test splits).
2. **Computes cell embeddings** using one of the supported models (Stack, scGPT, Geneformer, STATE, scConcept, SCSimilarity, etc.). Each model is an “extractor” that turns expression (or counts) into a fixed-size embedding per cell.
3. **Runs evaluations** on those embeddings. Evaluations depend on the task and can include:
   - **Batch integration**: batch-effect metrics (e.g. k-NN LISI), biological signal preservation, annotation (e.g. k-NN accuracy on cell type).
   - **Downstream tasks**: e.g. cancer subtype classification, treatment response, survival prediction, drug response (Depmap), so you can compare models on real benchmarks.

You choose a **YAML config** that specifies: which dataset, which embedding method, and which evaluations to run. Outputs (embeddings, metrics, plots) are written to a central output directory. The same config layout supports both single runs and parallel batch runs over many configs.

## Configuration

All paths are driven by environment variables so the codebase has no hardcoded absolute paths. Use a `.env` file in the repo root (or export variables in your shell).

### Required variables

Set these in `.env` (see `.env.example` for a template):

| Variable | Description |
|----------|-------------|
| `VCC_DATA` | Root path for datasets (e.g. where `DATASETS/`, etc. live). |
| `VCC_CHECKPOINTS_BASE` | Root path for model checkpoints (STACK, scGPT, Geneformer, etc.). |
| `SCFM_EVAL_OUTPUT_PATH` | Directory where all experiment outputs are written. |

### Optional variables

| Variable | Description |
|----------|-------------|
| `STACK_REPO_ROOT` | Path to the Stack repo (for the stack extractor). |
| `STATE_REPO_PATH` | Path to the STATE repo (for `pip install -e` in extractor env). |
| `SCGPT_REPO_PATH` | Path to the scGPT repo. |
| `GENEFORMER_REPO_PATH` | Path to the Geneformer repo. |
| `UCE_REPO_PATH` | Path to the UCE repo (used in PYTHONPATH). |
| `SCFM_EVAL_PARAMS_PATH` | Override directory for YAML configs (default: `yaml/` in repo). |

If you have a reference setup with concrete paths, copy from `dotenv.txt` into `.env` and adjust as needed. The app loads `.env` via `python-dotenv` on startup.

## Quick start

```bash
# 1. Copy env template and set your paths
cp .env.example .env
# Edit .env with VCC_DATA, VCC_CHECKPOINTS_BASE, SCFM_EVAL_OUTPUT_PATH (and optional repo paths)

# 2. Install dependencies (mamba/conda or pip as per your setup)

# 3. Run a single experiment from a YAML config
python run/run_exp.py yaml/batch_bio_integration/stack/cancer_TME/bassez_pre.yaml
```

YAML configs can use environment variable expansion (e.g. `${VCC_CHECKPOINTS_BASE}/STACK/bc_large_aligned.ckpt`). Values are expanded when configs are loaded.

## Running experiments

- **Single run:**  
  `python run/run_exp.py <config_path>`  
  Example: `python run/run_exp.py yaml/cancer_survival/pca/tcga_outcomes.yaml`

- **Parallel / batch:**  
  Run many configs in parallel with `python -m parallel_experiments.main`. You can restrict **which** configs run and **which GPUs** to use:

  | Option | Meaning | Example |
  |--------|---------|--------|
  | `-w`, `--workers` | Number of parallel workers (default: 4) | `-w 8` |
  | `--tasks` | Only run configs from these task directories (comma-separated). Task = top-level folder under `yaml/` (e.g. `batch_bio_integration`, `cancer_survival`, `drug_response`). | `--tasks drug_response,cancer_survival` |
  | `-m`, `--method` | Only run this embedding method (e.g. `stack`, `scgpt`, `scconcept`, `pca`). | `-m scconcept` |
  | `-d`, `--dataset` | Only run configs for this dataset (matches YAML filename / `dataset_name`). | `-d perturbseq_competition` |
  | `-s`, `--subgroup` | Only run configs under `task/method/<subgroup>/*.yaml` (e.g. `cancer_TME` under batch_bio_integration). | `--subgroup cancer_TME` |
  | `--gpus` | GPU IDs to use. Sets `CUDA_VISIBLE_DEVICES` so only these GPUs are visible to workers (e.g. `0`, `0,1`, `2,3`). | `--gpus 0,1` |
  | `--force` | Rerun even if results already exist (default: skip completed). | `--force` |
  | `--dry-run` | Print which configs would be run without executing. | `--dry-run` |

  **Examples:**

  ```bash
  # All new experiments, 8 workers
  python -m parallel_experiments.main -w 8

  # Only drug_response and cancer_survival tasks, 4 workers, use GPUs 0 and 1
  python -m parallel_experiments.main --tasks drug_response,cancer_survival -w 4 --gpus 0,1

  # Only batch_bio_integration, only configs under cancer_TME (bassez, chan2021, etc.)
  python -m parallel_experiments.main --tasks batch_bio_integration --subgroup cancer_TME -w 4 --gpus 0,1

  # Only scconcept on dataset perturbseq_competition, force rerun
  python -m parallel_experiments.main -m scconcept -d perturbseq_competition --force -w 2

  # See what would run (no execution)
  python -m parallel_experiments.main --tasks cancer_survival --dry-run
  ```

  By default, the runner **skips** experiments that already have results in the output directory. Use `--force` to rerun them. For more options (e.g. `--yaml-dir`, `--output-dir`), run `python -m parallel_experiments.main --help`.

### Structure of configs and results

Configs and outputs follow the same **nested layout**:

- **Task** — top-level benchmark (e.g. `batch_bio_integration`, `cancer_survival`, `drug_response`). Each task has its own evaluation types and datasets.
- **Model (method)** — embedding method (e.g. `stack`, `scgpt`, `scconcept`, `pca`). Under each task you have one folder per method.
- **Subgroup** — optional: some tasks group datasets into subgroups (e.g. `cancer_TME`, `technical_repeats` under `batch_bio_integration`). So you get `task/method/subgroup/` when present.
- **Dataset** — one YAML per dataset (e.g. `bassez_pre.yaml`, `tcga_outcomes.yaml`). Each YAML is one experiment; results are written under the same path.

So paths look like:

- `yaml/<task>/<method>/<dataset>.yaml` or `yaml/<task>/<method>/<subgroup>/<dataset>.yaml`
- Results: `OUTPUT_PATH/<task>/<method>/<dataset>/` or `OUTPUT_PATH/<task>/<method>/<subgroup>/<dataset>/` (each run produces metrics, embeddings, config copy, etc.)

The parallel runner’s `--tasks`, `-m`, `-d`, and `--subgroup` options filter exactly along this hierarchy.

### Results summarizer

Given that hierarchy, the **results summarizer** (`utils/results_summarizer.py`) **brings together results at each level** and compares them. It reads per-dataset metrics (e.g. `all_results_raw.csv` under each `task/method/[subgroup/]dataset/`) and, at every level, produces **comparison tables and boxplots** across the children:

- **Subgroup-level** (if present): compare datasets within `task/method/subgroup/` → writes to `task/method/subgroup/summaries/`
- **Method-level**: compare all datasets (and subgroups) for that method in the task → `task/method/summaries/`
- **Task-level**: compare all methods and datasets for the task → `task/summaries/`
- **Cross-task**: compare all tasks, methods, and datasets → `OUTPUT_PATH/summaries/`

So you get method-vs-method and dataset-vs-dataset comparisons with tables and boxplots at each scope. Metric values stay in their **original range and direction**; a composite **global_score** (mean of normalized metrics, see `utils/metric_definitions.py`) is used for ranking.

The summarizer runs **automatically after each experiment**. You can turn it off with **`SKIP_SUMMARIES=1`** (env) or **`dataset.skip_summaries: true`** in the YAML for that run. To run it manually on an existing output tree (e.g. after many parallel runs):

```bash
python -m utils.results_summarizer $SCFM_EVAL_OUTPUT_PATH
python -m utils.results_summarizer $SCFM_EVAL_OUTPUT_PATH --no-plots   # tables only, no boxplots
python -m utils.results_summarizer $SCFM_EVAL_OUTPUT_PATH/batch_bio_integration --scope task
```

Config files live under `yaml/` by default (or under `SCFM_EVAL_PARAMS_PATH` if set). Each YAML defines dataset, embedding method, and evaluations.

## Writing and reading a YAML config

Each experiment is defined by a single YAML file with these top-level keys:

| Key | Purpose |
|-----|--------|
| `run_id` | Optional identifier for this run (e.g. for logging); can be `null`. |
| `dataset` | How to load the data: loader class, path to the `.h5ad`, and metadata (label key, batch key, filters, train/test split, etc.). |
| `qc` | Quality control: `min_genes`, `min_cells`, or `skip: true` to disable. |
| `preprocessing` | Normalization and transform: e.g. `normalize: true`, `target_sum: 10000`, `apply_log1p: true`, or `skip: true`. |
| `hvg` | Highly variable genes: `n_top_genes`, `batch_key`, `flavor` (e.g. `seurat`), or `skip: true`. |
| `embedding` | Which model to use and its parameters: `method` (e.g. `stack`, `pca`, `scgpt`), `module`/`class`, and `params` (model-specific). |
| `evaluations` | List of evaluation blocks. Each has `type`, `skip`, and `params` (e.g. `batch_effects`, `biological_signal`, `annotation`, `survival`, `drug_response`). |

**Paths in the YAML**

- **Dataset path** (`dataset.path`): Either relative to `VCC_DATA` (e.g. `DATASETS/SCFM_datasets/.../file.h5ad`) or absolute using a variable, e.g. `path: ${VCC_DATA}/DATASETS/.../file.h5ad`.
- **Model/checkpoint paths** (`embedding.params`): Use env vars so the same config works everywhere, e.g. `checkpoint_path: ${VCC_CHECKPOINTS_BASE}/STACK/bc_large_aligned.ckpt`. Any string value is expanded with `os.path.expandvars` when the config is loaded (so `$VAR` and `${VAR}` both work).


## Extractors

Embedding models (Stack, scGPT, Geneformer, STATE, UCE, etc.) are in `extractors/`. Many use **separate virtual environments** and **local repo paths** (set via the optional env vars above).

- Setup: `cd extractors && ./setup_envs.sh [model]`  
  (Ensure the repo path env vars are set or the script is updated to use them.)
- Details: [extractors/README.md](extractors/README.md)

## Requirements

- Python 3.10+
- Dependencies as in the project (e.g. `pyproject.toml` or `requirements.txt` if present).
- `python-dotenv` for loading `.env`.
