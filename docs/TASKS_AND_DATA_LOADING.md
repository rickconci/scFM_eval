# Classification Tasks: Data Loading, Filtering, and Label Identification

## Overview

This document explains all classification tasks across different datasets, how data is loaded and filtered, and how labels are identified for each task.

---

## 1. Task Organization by Directory Structure

Tasks are organized by **dataset** and **task name** in the YAML directory structure:

```
yaml/
├── brca_full/
│   ├── all_cells/          # Task: All cells, no specific filtering
│   ├── subtype/            # Task: Cancer subtype classification
│   ├── chemo/              # Task: Treatment cohort classification
│   ├── pre_post/           # Task: Pre vs Post chemo classification
│   ├── cell_type/          # Task: Cell type annotation
│   └── outcome/            # Task: Treatment response prediction
└── luad1/                  # LUAD dataset
    └── (stage classification)
```

---

## 2. BRCA Tasks

### Task 1: **Cancer Subtype Classification** (`brca_full/subtype/`)

**Goal**: Classify cancer subtype (ER+ vs TNBC)

**YAML Config**:
```yaml
dataset:
  path: DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad
  label_key: Cancer_type          # Column containing labels
  batch_key: donor_id
  filter: 
    - column: Cancer_type           # Filter 1: Keep only ER+ and TNBC
      values:
        - ER+
        - TNBC
    - column: timepoint             # Filter 2: Keep only Pre-treatment samples
      values:
        - Pre

classification:
  params:
    cls_level: patient              # Patient-level classification
    label_map:
      ER+: 0
      TNBC: 1
    train_funcs: [avg, mil, vote]
```

**Data Flow**:
1. **Load**: Load full BRCA dataset (209,126 cells × 32,088 genes)
2. **Filter Step 1**: Keep only cells where `Cancer_type` is `ER+` or `TNBC`
   - Removes: `HER2+` cells
3. **Filter Step 2**: Keep only cells where `timepoint` is `Pre`
   - Removes: Post-treatment cells
4. **Label**: Use `Cancer_type` column as labels
5. **Classification**: Patient-level (aggregate cells per patient → predict subtype)

**Available Labels in Dataset**:
- `ER+`: 43,114 cells
- `TNBC`: 35,212 cells  
- `HER2+`: 9,000 cells (filtered out)

**Split Files**:
- `data_splits/brca_full/brca_subtype/train_test_split.json`
- `data_splits/brca_full/brca_subtype/cv_splits.json`

---

### Task 2: **Treatment Cohort Classification** (`brca_full/chemo/`)

**Goal**: Classify treatment cohort (treatment_naive vs neoadjuvant_chemo)

**YAML Config**:
```yaml
dataset:
  path: DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad
  label_key: cohort                 # Column containing labels
  batch_key: donor_id
  filter: 
    - column: timepoint             # Filter 1: Keep only Pre-treatment
      values:
        - Pre
    - column: cell_types            # Filter 2: Keep only Cancer cells
      values:
        - Cancer_cell

classification:
  params:
    cls_level: patient
    label_map:
      treatment_naive: 0
      neoadjuvant_chemo: 1
    train_funcs: [avg, mil, vote]
```

**Data Flow**:
1. **Load**: Full BRCA dataset
2. **Filter Step 1**: Keep only `timepoint == 'Pre'` cells
3. **Filter Step 2**: Keep only `cell_types == 'Cancer_cell'` cells
   - Removes: T cells, B cells, Fibroblasts, etc.
   - **Result**: Only cancer cells from pre-treatment samples
4. **Label**: Use `cohort` column (treatment_naive vs neoadjuvant_chemo)
5. **Classification**: Patient-level (aggregate cancer cells per patient → predict cohort)

**Available Labels in Dataset**:
- `treatment_naive`: All cells (87,326 cells total, but filtered to cancer cells only)
- `neoadjuvant_chemo`: Subset of patients

**Split Files**:
- `data_splits/brca_full/brca_chemo/train_test_split_oversampled.json`
- `data_splits/brca_full/brca_chemo/cv_splits_oversampled.json`

---

### Task 3: **Pre vs Post Chemo Classification** (`brca_full/pre_post/`)

**Goal**: Classify whether cells are from pre-treatment or post-treatment samples

**YAML Config**:
```yaml
dataset:
  path: DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad
  label_key: pre_post               # Column containing labels
  batch_key: donor_id
  # NO FILTERS - uses all cells from both timepoints

classification:
  params:
    cls_level: patient
    label_map:
      Pre: 0
      Post: 1
    train_funcs: [avg, mil, vote]
```

**Data Flow**:
1. **Load**: Full BRCA dataset
2. **No Filters**: Uses all cells from both Pre and Post timepoints
3. **Label**: Use `pre_post` column (`Pre` vs `Post`)
4. **Classification**: Patient-level (aggregate cells per patient → predict timepoint)

**Available Labels in Dataset**:
- `Pre`: 41,102 cells
- `Post`: 46,224 cells

**Split Files**:
- `data_splits/brca_full/brca_pre_post/train_test_split.json`
- `data_splits/brca_full/brca_pre_post/cv_splits.json`

---

### Task 4: **Cell Type Annotation** (`brca_full/cell_type/`)

**Goal**: Classify cell type for each individual cell

**YAML Config**:
```yaml
dataset:
  path: DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad
  label_key: cell_types             # Column containing labels
  batch_key: donor_id
  filter: 
    - column: timepoint             # Filter: Keep only Pre-treatment
      values:
        - Pre

classification:
  params:
    cls_level: cell                 # CELL-LEVEL classification (not patient-level!)
    label_map:
      T_cell: 0
      Cancer_cell: 1
      Fibroblast: 2
      Myeloid_cell: 3
      B_cell: 4
      Endothelial_cell: 5
      Mast_cell: 6
      pDC: 7
    # train_funcs not used for cell-level
```

**Data Flow**:
1. **Load**: Full BRCA dataset
2. **Filter**: Keep only `timepoint == 'Pre'` cells
3. **Label**: Use `cell_types` column for each cell
4. **Classification**: **Cell-level** (direct classification of each cell, no aggregation)

**Available Labels in Dataset** (after filtering to Pre):
- `T_cell`: 27,007 cells
- `Cancer_cell`: 26,822 cells
- `Fibroblast`: 14,793 cells
- `Myeloid_cell`: 8,317 cells
- `B_cell`: 5,890 cells
- `Endothelial_cell`: 3,828 cells
- `Mast_cell`: 438 cells
- `pDC`: 231 cells

**Split Files**:
- `data_splits/brca_full/brca_cell_type/train_test_split.json`
- `data_splits/brca_full/brca_cell_type/cv_splits.json`

---

### Task 5: **Treatment Response Prediction** (`brca_full/outcome/`)

**Goal**: Predict treatment response (Non-Responder vs Responder)

**YAML Config**:
```yaml
dataset:
  path: mnt/DATA/brca_full/T_cells_only.h5ad  # DIFFERENT DATASET: T cells only!
  label_key: outcome                           # Column containing labels
  batch_key: donor_id
  filter: 
    - column: outcome                          # Filter 1: Keep only E and NE
      values:
        - NE                                  # Non-Responder
        - E                                   # Responder
    - column: timepoint                        # Filter 2: Keep only Pre-treatment
      values:
        - Pre

classification:
  params:
    cls_level: patient
    label_map:
      NE: 0                                   # Non-Responder
      E: 1                                    # Responder
    train_funcs: [avg, mil, vote]
```

**Data Flow**:
1. **Load**: **Different dataset** - `T_cells_only.h5ad` (only T cells, not full dataset)
2. **Filter Step 1**: Keep only cells where `outcome` is `NE` or `E`
   - Removes: `n/a` (6,000 cells)
3. **Filter Step 2**: Keep only `timepoint == 'Pre'` cells
4. **Label**: Use `outcome` column
5. **Classification**: Patient-level (aggregate T cells per patient → predict response)

**Available Labels in Dataset**:
- `NE` (Non-Responder): 54,326 cells
- `E` (Responder): 27,000 cells
- `n/a`: 6,000 cells (filtered out)

**Split Files**:
- `data_splits/brca_full/brca_outcome/train_test_split_oversampled.json`
- `data_splits/brca_full/brca_outcome/cv_splits_oversampled.json`

---

### Task 6: **All Cells** (`brca_full/all_cells/`)

**Goal**: General evaluation on all cells (no specific classification task)

**YAML Config**:
```yaml
dataset:
  path: DATASETS/SCFM_datasets/Bassez_2021_processed/bassez_combined_preprocessed_prepared.h5ad
  label_key: cell_types
  batch_key: donor_id
  filter: 
    - column: timepoint
      values:
        - Pre
  # No classification task - just embedding extraction and evaluation
```

**Data Flow**:
1. **Load**: Full BRCA dataset
2. **Filter**: Keep only `timepoint == 'Pre'` cells
3. **No Classification**: Used for embedding extraction and batch effects/biological signal evaluation only

---

## 3. LUAD Tasks

### Task: **Cancer Stage Classification** (`luad1/`, `luad2/`)

**Goal**: Classify cancer stage (early_stage vs late_stage)

**YAML Config**:
```yaml
dataset:
  path: mnt/DATA/luad_brca/luad/luad_analysis1.h5ad  # LUAD dataset
  label_key: group_id                                # Column containing labels
  batch_key: sample_id
  # NO FILTERS - uses all cells

classification:
  params:
    cls_level: patient
    label_map:
      early_stage: 0
      late_stage: 1
    train_funcs: [avg, mil, vote]
```

**Data Flow**:
1. **Load**: LUAD dataset (different from BRCA)
2. **No Filters**: Uses all cells
3. **Label**: Use `group_id` column
4. **Classification**: Patient-level (aggregate cells per sample → predict stage)

**Available Labels in Dataset**:
- `early_stage`: 10,394 cells
- `late_stage`: 6,400 cells

**Split Files**:
- `data_splits/luad/luad1/train_test_split.json`
- `data_splits/luad/luad1/cv_splits.json`

**Note**: There are two LUAD datasets:
- `luad1`: Uses `luad_analysis1.h5ad`
- `luad2`: Uses `luad_analysis2.h5ad` (may have different filters or labels)

---

## 4. How Filters Work

### Filter Configuration

Filters are specified as a list in the YAML:
```yaml
filter: 
  - column: timepoint
    values:
      - Pre
  - column: cell_types
    values:
      - Cancer_cell
```

### Filter Application Logic

**Code Location**: `data/data_loader.py::_filter()`

**How it works**:
1. **Load data** first (full dataset)
2. **Build combined mask**: For each filter, create a boolean mask
3. **Combine masks**: Use AND logic (cell must match ALL filters)
4. **Apply filter**: Return filtered AnnData view

```python
def _filter(self, adata, filter_dict):
    """Apply filters to AnnData object.
    
    Args:
        filter_dict: Dictionary of {column: [values]} to filter by.
    
    Returns:
        Filtered AnnData object.
    """
    # Build combined mask (AND logic: cell must match ALL filters)
    mask = np.ones(adata.n_obs, dtype=bool)
    for col, values in filter_dict.items():
        mask &= adata.obs[col].isin(values).values
    
    return adata[mask]
```

**Example**:
- Filter 1: `timepoint in ['Pre']` → mask1
- Filter 2: `cell_types in ['Cancer_cell']` → mask2
- Combined: `mask = mask1 & mask2` (cells must be Pre AND Cancer_cell)

---

## 5. How Labels Are Identified

### Label Key Configuration

The `label_key` in the YAML specifies which column in `adata.obs` contains the labels:

```yaml
dataset:
  label_key: Cancer_type    # Uses adata.obs['Cancer_type'] as labels
```

### Label Mapping

For classification, labels are mapped from strings to integers:

```yaml
classification:
  params:
    label_map:
      ER+: 0
      TNBC: 1
```

**Code Flow**:
1. Extract labels: `adata.obs[label_key]` → string labels (e.g., `'ER+'`, `'TNBC'`)
2. Map to integers: `label_map['ER+']` → `0`, `label_map['TNBC']` → `1`
3. Train classifier: Uses integer labels (0, 1, ...)

---

## 6. Complete Task Summary Table

| Task | Dataset | Label Key | Filters | Classification Level | Label Map |
|------|---------|-----------|---------|---------------------|-----------|
| **Subtype** | BRCA | `Cancer_type` | `timepoint=Pre`, `Cancer_type in [ER+, TNBC]` | Patient | ER+:0, TNBC:1 |
| **Chemo** | BRCA | `cohort` | `timepoint=Pre`, `cell_types=Cancer_cell` | Patient | treatment_naive:0, neoadjuvant_chemo:1 |
| **Pre/Post** | BRCA | `pre_post` | None | Patient | Pre:0, Post:1 |
| **Cell Type** | BRCA | `cell_types` | `timepoint=Pre` | **Cell** | T_cell:0, Cancer_cell:1, ... |
| **Outcome** | BRCA (T cells) | `outcome` | `timepoint=Pre`, `outcome in [NE, E]` | Patient | NE:0, E:1 |
| **Stage** | LUAD | `group_id` | None | Patient | early_stage:0, late_stage:1 |

---

## 7. Data Loading Pipeline

### Step-by-Step Process

1. **Load Dataset** (`H5ADLoader.load()`):
   - Read `.h5ad` file
   - Load into `AnnData` object
   - Shape: `(n_cells, n_genes)`

2. **Apply Filters** (`H5ADLoader._filter()`):
   - For each filter entry: `{column: [values]}`
   - Create boolean mask: `adata.obs[column].isin(values)`
   - Combine masks with AND logic
   - Return filtered AnnData: `adata[mask]`

3. **Quality Control** (`H5ADLoader.qc()`):
   - Filter cells: `min_genes` per cell
   - Filter genes: `min_cells` per gene

4. **Preprocessing** (`H5ADLoader.scale()`):
   - Normalize: `normalize_total(target_sum=10000)`
   - Log transform: `log1p()`

5. **HVG Selection** (`H5ADLoader.select_hvg()`):
   - Select highly variable genes
   - Reduce to top N genes (e.g., 4096)

6. **Extract Embeddings**:
   - Run embedding method (e.g., SCimilarity, scConcept)
   - Store in `adata.obsm[embedding_key]`

7. **Classification**:
   - Extract labels: `adata.obs[label_key]`
   - Map labels: `label_map[string_label]` → integer
   - Train classifier on embeddings

---

## 8. Key Observations

### Different Datasets for Different Tasks

- **Most tasks**: Use `bassez_combined_preprocessed_prepared.h5ad` (full dataset)
- **Outcome task**: Uses `T_cells_only.h5ad` (pre-filtered to T cells only)
- **LUAD tasks**: Use `luad_analysis1.h5ad` or `luad_analysis2.h5ad`

### Filter Strategy

- **Subtype, Chemo, Cell Type**: Filter to `timepoint=Pre` (pre-treatment only)
- **Pre/Post**: No filter (uses both timepoints)
- **Outcome**: Filter to `timepoint=Pre` AND `outcome in [NE, E]`

### Classification Level

- **Patient-level** (most tasks): Aggregate cells per patient → predict patient label
- **Cell-level** (cell_type task): Direct cell classification

### Label Columns

Different tasks use different columns:
- `Cancer_type`: ER+, TNBC, HER2+
- `cohort`: treatment_naive, neoadjuvant_chemo
- `pre_post`: Pre, Post
- `cell_types`: T_cell, Cancer_cell, Fibroblast, etc.
- `outcome`: NE, E, n/a
- `group_id`: early_stage, late_stage

---

## 9. Example: Complete Flow for Subtype Task

```python
# 1. Load
adata = ad.read_h5ad("bassez_combined_preprocessed_prepared.h5ad")
# Shape: (209126, 32088)

# 2. Filter
# Filter 1: Cancer_type in [ER+, TNBC]
mask1 = adata.obs['Cancer_type'].isin(['ER+', 'TNBC'])
# Filter 2: timepoint == 'Pre'
mask2 = adata.obs['timepoint'] == 'Pre'
# Combined
adata = adata[mask1 & mask2]
# Shape: ~(78,326, 32088)  # Approximate after filtering

# 3. QC
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)

# 4. Preprocessing
sc.pp.normalize_total(adata, target_sum=10000)
sc.pp.log1p(adata)

# 5. HVG
sc.pp.highly_variable_genes(adata, n_top_genes=4096)
adata = adata[:, adata.var.highly_variable]

# 6. Extract Embeddings
embeddings = extractor.extract(adata)  # Shape: (n_cells, embedding_dim)
adata.obsm['X_scimilarity'] = embeddings

# 7. Classification
# Extract labels
labels = adata.obs['Cancer_type']  # ['ER+', 'TNBC', ...]
# Map to integers
label_map = {'ER+': 0, 'TNBC': 1}
y = labels.map(label_map)  # [0, 1, 0, ...]

# Aggregate per patient
patient_embeddings = aggregate_cells_per_patient(adata, embeddings)
patient_labels = get_patient_labels(adata, y)

# Train classifier
classifier.fit(patient_embeddings, patient_labels)
```

---

This document provides a complete overview of how tasks are defined, how data is loaded and filtered, and how labels are identified for each classification task.
