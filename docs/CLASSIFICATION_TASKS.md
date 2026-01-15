# Classification Tasks: Patient-Level vs Cell-Level

## Overview

The classification system supports two main **classification levels**:
1. **Patient/Sample-Level** (`cls_level: 'patient'`)
2. **Cell-Level** (`cls_level: 'cell'`)

Each level has different tasks and approaches. Let's break them down:

---

## 1. Patient-Level Classification (`cls_level: 'patient'`)

**Goal**: Predict a label for each **patient/sample** (not individual cells).

**Key Point**: Patients have multiple cells, so we need to aggregate cell-level embeddings into patient-level features.

### Configuration
```yaml
cls_level: patient
train_funcs: [avg, mil, vote]  # Choose which aggregation methods to use
```

### Three Different Tasks (Aggregation Methods):

#### Task 1: **Average Embedding** (`train_funcs: ['avg']`)

**How it works**:
1. For each patient, take all their cells' embeddings
2. **Average** the embeddings across all cells → one embedding vector per patient
3. Train classifier on patient-level averaged embeddings
4. Predict patient label

**Code Flow**:
```python
# In __train_avg_expression():
def aggregate_embeddings(adata):
    # Get embeddings for all cells
    emb = adata.obsm[embedding_col]  # Shape: (n_cells, embedding_dim)
    
    # Group by patient/sample_id
    sample_ids = adata.obs['sample_id'].values
    df_emb = pd.DataFrame(emb, index=sample_ids)
    
    # Average embeddings per patient
    mean_emb = df_emb.groupby(sample_ids).mean()  # Shape: (n_patients, embedding_dim)
    
    # Get patient labels
    labels = adata.obs[['sample_id', 'label']].drop_duplicates()
    
    return mean_emb, labels['label']

# Train classifier on patient-level features
X_train, y_train = aggregate_embeddings(adata_train)  # X_train: (n_patients, embedding_dim)
X_test, y_test = aggregate_embeddings(adata_test)
classifier.fit(X_train, y_train)  # Predicts patient labels
```

**Output**: Patient-level predictions
- `cls_metrics_avg_expr.csv`: Overall metrics (AUC, F1, etc.)
- `cls_per_class_metrics_avg_expr.csv`: Per-class metrics (precision, recall, F1 per label)

---

#### Task 2: **Multi-Instance Learning (MIL)** (`train_funcs: ['mil']`)

**How it works**:
1. Treat each patient as a "bag" of cells (instances)
2. Use attention mechanism to learn which cells are important
3. Weighted aggregation of cell embeddings based on attention
4. Train classifier on attention-weighted patient embeddings
5. Predict patient label

**Code Flow**:
```python
# In __train_mil():
from models.mil_experiment import MILExperiment

exp = MILExperiment(
    embedding_col=self.embedding_col,
    label_key='label',
    patient_key="sample_id"
)
exp.train(adata_train)  # Learns attention weights

# Evaluate: attention-weighted aggregation → patient prediction
pids, y_true, preds, pred_scores = exp.evaluate(adata_test)
```

**Key Difference from `avg`**: 
- `avg`: Simple mean (all cells weighted equally)
- `mil`: Attention-weighted mean (learns which cells matter more)

**Output**: Patient-level predictions with attention analysis
- `cls_metrics_mil.csv`: Overall metrics
- `cls_per_class_metrics_mil.csv`: Per-class metrics
- `attention_analysis.png`: Visualization of which cells get high attention
- `top_attention_cells.csv`: Top cells by attention weight

---

#### Task 3: **Majority Vote** (`train_funcs: ['vote']`)

**How it works**:
1. First, classify **each cell** individually (cell-level classification)
2. For each patient, collect all cell predictions
3. Take **majority vote** of cell predictions → patient prediction
4. Also average prediction scores per patient

**Code Flow**:
```python
# In __train_vote():
# Step 1: Classify each cell
cell_pred_train, cell_pred_test, _, _ = self.__train_cell(adata_train, adata_test)

# Step 2: Aggregate cell predictions to patient level
def save_patient_level(adata_subset):
    # Group by patient
    y_pred_p = adata_subset.groupby('sample_id')['pred'].agg(
        lambda x: x.value_counts().idxmax()  # Majority vote
    )
    y_pred_score_p = adata_subset.groupby('sample_id')['pred_score'].mean()  # Average score
    y_test_p = adata_subset.groupby('sample_id')['label'].first()
    
    return patient_predictions

pred_df_test = save_patient_level(cell_pred_test)
```

**Key Difference**:
- `avg`/`mil`: Aggregate embeddings → classify patient
- `vote`: Classify cells → aggregate predictions → patient label

**Output**: Patient-level predictions (derived from cell votes)
- `cls_metrics_vote.csv`: Overall metrics
- `cls_per_class_metrics_vote.csv`: Per-class metrics

---

### Summary: Patient-Level Tasks

| Task | Method | Aggregation | Prediction Level |
|------|--------|-------------|-----------------|
| `avg` | Average embeddings | Mean of cell embeddings | Patient |
| `mil` | Attention-weighted | Weighted mean (learned) | Patient |
| `vote` | Majority vote | Vote on cell predictions | Patient |

**All three produce patient-level predictions**, but use different strategies to aggregate cell-level information.

---

## 2. Cell-Level Classification (`cls_level: 'cell'`)

**Goal**: Predict a label for each **individual cell**.

**Key Point**: No aggregation needed - classify cells directly.

### Configuration
```yaml
cls_level: cell
# train_funcs not used (only one method)
```

### How it Works:

**Code Flow**:
```python
# In __train_cell():
# Extract embeddings and labels for each cell
X_train = adata_train.obsm[embedding_col]  # Shape: (n_cells_train, embedding_dim)
X_test = adata_test.obsm[embedding_col]   # Shape: (n_cells_test, embedding_dim)

y_train = adata_train.obs['label'].values  # Shape: (n_cells_train,)
y_test = adata_test.obs['label'].values    # Shape: (n_cells_test,)

# Train classifier directly on cell embeddings
classifier.fit(X_train, y_train)

# Predict each cell's label
y_pred_test = classifier.predict(X_test)  # One prediction per cell
```

**Output**: Cell-level predictions
- `cell_pred_test.csv`: Predictions for each cell (cell_id, label, pred, pred_score)
- `cls_metrics_cell.csv`: Overall metrics (AUC, F1, etc. across all cells)
- `cls_per_class_metrics_cell.csv`: Per-class metrics (precision, recall, F1 per label)

---

## 3. Data Splits

Both patient-level and cell-level classification support:

### Single Split (`onesplit: True`)
- One train/test split
- Train on train set, evaluate on test set
- Results: Single set of metrics

### Cross-Validation (`cv: True` or `cv: 'loocv'`)
- Multiple folds (e.g., 5-fold CV)
- Train on each fold's train set, evaluate on fold's test set
- Results: 
  - Per-fold metrics: `{method}_cv_metrics.csv`
  - **Aggregated metrics**: `{method}_cv_metrics_aggregated.csv` (mean ± std across folds)
  - **Aggregated per-class**: `{method}_cv_per_class_metrics_aggregated.csv` (mean ± std per class)

---

## 4. Complete Example

### Scenario: Predict treatment response (binary: treatment_naive vs neoadjuvant_chemo)

**Config**:
```yaml
cls_level: patient
train_funcs: [avg, mil, vote]
cv: True
onesplit: True
label_map:
  treatment_naive: 0
  neoadjuvant_chemo: 1
```

**What happens**:
1. **Single Split** (`onesplit=True`):
   - For each method (`avg`, `mil`, `vote`):
     - Train on train patients
     - Predict test patients
     - Save metrics

2. **Cross-Validation** (`cv=True`):
   - For each method (`avg`, `mil`, `vote`):
     - For each fold (e.g., 5 folds):
       - Train on fold's train patients
       - Predict fold's test patients
     - Aggregate metrics across folds (mean ± std)

**Output Files**:
```
metrics/classification/
├── cls_metrics_avg_expr.csv                    # Single split: avg method
├── cls_per_class_metrics_avg_expr.csv          # Single split: avg per-class
├── cls_metrics_mil.csv                         # Single split: mil method
├── cls_per_class_metrics_mil.csv                # Single split: mil per-class
├── cls_metrics_vote.csv                        # Single split: vote method
├── cls_per_class_metrics_vote.csv              # Single split: vote per-class
└── cv/
    ├── avg_cv_metrics.csv                       # CV: avg (all folds)
    ├── avg_cv_metrics_aggregated.csv            # CV: avg (mean ± std)
    ├── avg_cv_per_class_metrics_aggregated.csv  # CV: avg per-class (mean ± std)
    ├── mil_cv_metrics.csv
    ├── mil_cv_metrics_aggregated.csv
    ├── mil_cv_per_class_metrics_aggregated.csv
    ├── vote_cv_metrics.csv
    ├── vote_cv_metrics_aggregated.csv
    └── vote_cv_per_class_metrics_aggregated.csv
```

---

## 5. Key Differences Summary

| Aspect | Patient-Level | Cell-Level |
|--------|---------------|------------|
| **Prediction Target** | Patient/Sample | Individual Cell |
| **Input Features** | Aggregated (avg/mil/vote) | Direct cell embeddings |
| **Number of Predictions** | n_patients | n_cells |
| **Tasks Available** | 3 (avg, mil, vote) | 1 (direct) |
| **Use Case** | Clinical outcomes, treatment response | Cell type classification, cell state |

---

## 6. When to Use Which?

### Use Patient-Level (`cls_level: 'patient'`) when:
- You want to predict **patient outcomes** (treatment response, survival, disease status)
- Labels are at the **patient level** (e.g., "responder" vs "non-responder")
- You have multiple cells per patient and want to leverage all of them

### Use Cell-Level (`cls_level: 'cell'`) when:
- You want to predict **cell properties** (cell type, cell state, cell condition)
- Labels are at the **cell level** (e.g., "T cell" vs "B cell")
- You want to understand individual cell predictions

---

## 7. Per-Class Metrics

**Both levels** compute per-class metrics:
- **Per-class precision**: How accurate are predictions for each class?
- **Per-class recall**: How many true positives are found for each class?
- **Per-class F1**: Harmonic mean of precision and recall
- **Support**: Number of samples/cells in each class

These are saved in:
- Single split: `{model}_per_class_metrics_{method}.csv`
- CV aggregated: `{method}_cv_per_class_metrics_aggregated.csv` (mean ± std)

Example for binary classification:
```
Metrics,random_forest
treatment_naive_precision,0.85
treatment_naive_recall,0.90
treatment_naive_f1,0.87
treatment_naive_support,150
neoadjuvant_chemo_precision,0.88
neoadjuvant_chemo_recall,0.82
neoadjuvant_chemo_f1,0.85
neoadjuvant_chemo_support,120
```
