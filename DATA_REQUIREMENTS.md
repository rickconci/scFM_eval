# Data Requirements for Running `run_brca.sh`

This document outlines the data requirements needed to run the `run_brca.sh` evaluation script.

## 1. Directory Structure

The script expects the following directory structure (defined in `setup_path.py`):

```
DATA_PATH = '/home/jupyter'  # Base data directory
├── mnt/
│   └── DATA/
│       └── brca_full/
│           └── brca_cells_only_3000cell_4096gene.h5ad  # Main data file

BASE_PATH = <scFM_eval root>  # Base path for configs and splits
├── data_splits/
│   └── brca_full/
│       ├── brca_cell_type/
│       │   ├── train_test_split.json
│       │   └── cv_splits.json
│       ├── brca_chemo/
│       ├── brca_outcome/
│       ├── brca_pre_post/
│       └── brca_subtype/
└── yaml/
    └── brca_full/
        ├── cell_type/
        ├── chemo/
        ├── outcome/
        ├── pre_post/
        └── subtype/
```

## 2. AnnData File Requirements

### File Format
- **Format**: H5AD (AnnData format)
- **Location**: `{DATA_PATH}/mnt/DATA/brca_full/brca_cells_only_3000cell_4096gene.h5ad`
- The path in YAML is relative to `DATA_PATH` (e.g., `mnt/DATA/brca_full/...`)

### Required AnnData Structure

#### `adata.X` (Expression Matrix)
- **Shape**: `(n_cells, n_genes)`
- **Type**: Sparse or dense matrix
- **Values**: Raw counts or normalized expression
- The script will normalize/log-transform if specified in config

#### `adata.obs` (Cell Metadata) - **REQUIRED COLUMNS**

1. **`cell_types`** (or `label_key` from config)
   - **Purpose**: Cell type labels for classification
   - **Type**: Categorical or string
   - **Example values**: `['T_cell', 'Cancer_cell', 'Fibroblast', ...]`
   - **Used for**: Classification labels

2. **`donor_id`** (or `batch_key` from config)
   - **Purpose**: Patient/donor identifier for train/test splitting
   - **Type**: String or categorical
   - **Critical**: Used for patient-level splits to avoid data leakage
   - **Used for**: Stratified train/test splits and cross-validation

3. **`timepoint`** (optional, for filtering)
   - **Purpose**: Filter cells by timepoint (e.g., 'Pre', 'Post')
   - **Used in**: Filter configuration in YAML

#### `adata.var` (Gene Metadata) - **REQUIRED FOR scConcept**

1. **`gene_ids`** (for scConcept)
   - **Purpose**: Gene identifiers in ENSG format
   - **Format**: ENSEMBL gene IDs (e.g., `ENSG00000139618`)
   - **Type**: String
   - **Required for**: scConcept embedding extraction
   - **Note**: If not present, scConcept will use `adata.var.index` as fallback

#### `adata.layers` (Optional)
- **`counts`**: Raw count matrix (if different from `adata.X`)
- **`X`**: Alternative expression layer (specified via `layer_name` in config)

## 3. Data Split Files (JSON)

### Train/Test Split (`train_test_split.json`)

**Location**: `{BASE_PATH}/data_splits/brca_full/{task}/train_test_split.json`

**Structure**:
```json
{
  "train_test_split": {
    "train_ids": ["donor_1", "donor_2", ...],
    "test_ids": ["donor_3", "donor_4", ...],
    "train_labels": ["T_cell", "Cancer_cell", ...],
    "test_labels": ["T_cell", "Fibroblast", ...]
  },
  "id_column": "donor_id",
  "label_column": "cell_types",
  "random_state": 42,
  "test_size": 0.33,
  "oversampled": false
}
```

**Requirements**:
- `train_ids` and `test_ids`: Lists of patient/donor IDs (must match `adata.obs[batch_key]`)
- Split is done at **patient level**, not cell level (prevents data leakage)
- Stratified by labels to maintain class balance

### Cross-Validation Splits (`cv_splits.json`)

**Location**: `{BASE_PATH}/data_splits/brca_full/{task}/cv_splits.json`

**Structure**:
```json
{
  "fold_1": {
    "train_ids": ["donor_1", "donor_2", ...],
    "test_ids": ["donor_5", ...],
    "train_labels": ["T_cell", ...],
    "test_labels": ["Cancer_cell", ...]
  },
  "fold_2": { ... },
  ...
  "id_column": "donor_id",
  "label_column": "cell_types",
  "random_state": 42,
  "n_splits": 5
}
```

**Requirements**:
- 5-fold cross-validation by default
- Each fold contains train/test splits at patient level
- Stratified to maintain class balance

## 4. Task-Specific Requirements

The script runs multiple tasks, each with different label mappings:

### Cell Type Classification (`cell_type`)
- **Label column**: `cell_types`
- **Label map**:
  ```yaml
  T_cell: 0
  Cancer_cell: 1
  Fibroblast: 2
  Myeloid_cell: 3
  B_cell: 4
  Endothelial_cell: 5
  Mast_cell: 6
  pDC: 7
  ```

### Pre vs Post (`pre_post`)
- **Label column**: `timepoint` or similar
- **Filter**: Only 'Pre' timepoint cells

### Subtype (`subtype`)
- **Label column**: Subtype labels (e.g., ER+ vs TNBC)

### Outcome (`outcome`)
- **Label column**: Outcome labels (e.g., E vs NE)

### Chemo (`chemo`)
- **Label column**: Treatment labels (e.g., chemo vs naive)

## 5. Preprocessing Pipeline

The data goes through these steps (configured in YAML):

1. **Quality Control** (`qc`)
   - Filter cells: `min_genes` (default: 10)
   - Filter genes: `min_cells` (default: 10)

2. **Preprocessing** (`preprocessing`)
   - Normalize: `normalize_total(target_sum=10000)`
   - Log transform: `log1p()`

3. **HVG Selection** (optional, for some models)
   - Select top N highly variable genes
   - Flavor: 'seurat'

## 6. Creating Data Splits

If you don't have split files, create them using `data_splits/split_data.py`:

```python
from data_splits.split_data import save_train_test_split, save_cv_splits
import anndata as ad

# Load your data
adata = ad.read_h5ad('path/to/your/data.h5ad')

# Get patient info
patient_ids = adata.obs['donor_id'].values
labels = adata.obs['cell_types'].values
counts = adata.obs['donor_id'].value_counts().values

# Create splits
save_dir = 'data_splits/brca_full/brca_cell_type'
save_train_test_split(
    patient_ids, labels, counts,
    save_dir=save_dir,
    patient_key='donor_id',
    label_key='cell_types',
    test_size=0.33,
    random_state=42
)

save_cv_splits(
    patient_ids, labels,
    save_dir=save_dir,
    patient_key='donor_id',
    label_key='cell_types',
    n_splits=5,
    random_state=42
)
```

## 7. Quick Checklist

Before running `bash run/run_brca.sh`, ensure:

- [ ] H5AD file exists at `{DATA_PATH}/mnt/DATA/brca_full/brca_cells_only_3000cell_4096gene.h5ad`
- [ ] `adata.obs['cell_types']` exists with cell type labels
- [ ] `adata.obs['donor_id']` exists with patient/donor IDs
- [ ] `adata.var['gene_ids']` exists (for scConcept) with ENSG format IDs
- [ ] Data split JSON files exist in `data_splits/brca_full/{task}/`
- [ ] Split files contain patient IDs that match `adata.obs['donor_id']`
- [ ] Label values in split files match `adata.obs['cell_types']`
- [ ] `DATA_PATH` in `setup_path.py` points to correct base directory
- [ ] scConcept model cache directory exists: `/lotterlab/datasets/VCC/SC_FM_repo_checkpoints/scConcept`

## 8. Common Issues

### Issue: "KeyError: 'donor_id'"
**Solution**: Ensure `adata.obs` has the column specified in `batch_key` (default: `donor_id`)

### Issue: "KeyError: 'cell_types'"
**Solution**: Ensure `adata.obs` has the column specified in `label_key` (default: `cell_types`)

### Issue: "Patient IDs in split file not found in data"
**Solution**: Verify that patient IDs in JSON files exactly match values in `adata.obs[batch_key]`

### Issue: "scConcept: gene_ids not found"
**Solution**: Add `gene_ids` column to `adata.var` with ENSG format IDs, or update `gene_id_column` in YAML

### Issue: "Data path not found"
**Solution**: Update `DATA_PATH` in `setup_path.py` or adjust path in YAML configs

