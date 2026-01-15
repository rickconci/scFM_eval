# Classification System Deep Dive

## Overview

The classification system trains and evaluates classifiers on embeddings to predict labels (e.g., cell types, treatment response). It supports multiple training strategies, cross-validation, and different classification levels (patient vs. cell).

---

## 1. Configuration (YAML)

### Location in Config
```yaml
evaluations:
  - type: classification
    skip: False
    params:
      module: models.classify
      class: ClassifierPipeline
      model: randomforest          # Classifier type: randomforest, logistic_regression, svc
      n_estimators: 100            # Model hyperparameters
      max_depth: 5
      train_funcs:                 # Training strategies to use
        - avg                      # Average embedding per sample
        - mil                      # Multi-instance learning
        - vote                     # Majority vote from cell predictions
      cv: True                     # Enable cross-validation (True, False, or 'loocv')
      onesplit: False              # Also run single train/test split
      label_map:                   # Map string labels to integers
        treatment_naive: 0
        neoadjuvant_chemo: 1
      cls_level: patient           # Classification level: 'patient' or 'cell'
      viz: True                    # Generate plots
      eval: True                   # Evaluate and save metrics
```

### Key Parameters

- **`model`**: Classifier type (RandomForest, LogisticRegression, SVC)
- **`train_funcs`**: List of training strategies:
  - `avg`: Average embeddings per sample, then classify
  - `mil`: Multi-instance learning (attention-based)
  - `vote`: Cell-level predictions → majority vote per sample
- **`cv`**: Cross-validation mode:
  - `True`: Use CV splits from `cv_splits` JSON file
  - `'loocv'`: Leave-one-out CV
  - `False`: No CV (only single split if `onesplit=True`)
- **`onesplit`**: If `True`, also run single train/test split
- **`cls_level`**: 
  - `'patient'`: Sample-level classification (uses `train_funcs`)
  - `'cell'`: Direct cell-level classification
- **`label_map`**: Maps string labels to integers (required for training)

### Data Splits

Splits are loaded from JSON files specified in the dataset config:
```yaml
dataset:
  train_test_split: data_splits/brca_full/brca_cell_type/train_test_split.json
  cv_splits: data_splits/brca_full/brca_cell_type/cv_splits.json
```

**train_test_split.json** structure:
```json
{
  "id_column": "patient_id",
  "train_test_split": {
    "train_ids": ["patient_1", "patient_2", ...],
    "test_ids": ["patient_3", "patient_4", ...]
  }
}
```

**cv_splits.json** structure:
```json
{
  "id_column": "patient_id",
  "n_splits": 5,
  "fold_1": {
    "train_ids": ["patient_1", "patient_2", ...],
    "test_ids": ["patient_3", ...]
  },
  "fold_2": { ... },
  ...
}
```

---

## 2. Execution Flow

### Step 1: EvaluationRunner.run_classification_evaluation()

**Location**: `run/evaluation_runner.py:203`

1. Extracts classification params from config
2. Creates `ClassifierPipeline` instance
3. Calls `clf.train(loader)` with the data loader

**Key code**:
```python
clf_params['save_dir'] = self.metrics_classification_dir
clf_params['plots_dir'] = self.plots_evaluations_dir
clf_params['embedding_col'] = self.embedding_key  # e.g., 'X_scimilarity'
ClfClass = load_class_func(module, class_name)
clf = ClfClass(clf_config)
clf.train(loader)
```

### Step 2: ClassifierPipeline.train()

**Location**: `models/classify.py:222`

Routes to appropriate training method based on `cls_level`:

- **`cls_level == 'patient'`** → `train_sample()` (patient-level)
- **`cls_level == 'cell'`** → `__train_cell()` (cell-level)

### Step 3: Patient-Level Training (`train_sample()`)

**Location**: `models/classify.py:241`

1. **Label Encoding**: Maps string labels to integers using `label_map`
2. **Select Training Functions**: Based on `train_funcs` config
3. **Single Split** (if `onesplit=True`):
   - Loads train/test split from `train_test_split.json`
   - For each training function (`avg`, `mil`, `vote`):
     - Trains classifier
     - Evaluates on test set
     - Saves results
4. **Cross-Validation** (if `cv=True` or `cv='loocv'`):
   - Loads CV splits from `cv_splits.json`
   - For each training function:
     - Runs `__train_cv()` which trains on each fold
     - Aggregates predictions/metrics across folds
     - Saves CV results

### Step 4: Training Functions

#### A. `__train_avg_expression()` - Average Embedding
**Location**: `models/classify.py:380`

**Process**:
1. Groups cells by `sample_id`
2. Averages embeddings per sample → `X_train`, `X_test` (sample-level)
3. Extracts labels per sample
4. Trains classifier on sample-level features
5. Evaluates and saves results

**Output files**:
- `cls_predictions_avg_expr.csv`
- `cls_metrics_avg_expr.csv`
- `cls_report_avg_expr.csv`
- `cls_metrics_avg_expr.png` (plots)

#### B. `__train_mil()` - Multi-Instance Learning
**Location**: `models/classify.py:493`

**Process**:
1. Uses `MILExperiment` class (attention-based MIL)
2. Trains on cell-level embeddings with sample-level labels
3. Attention mechanism learns which cells are important
4. Evaluates on test set
5. If `viz=True`, generates attention visualizations

**Output files**:
- `cls_predictions_mil.csv`
- `cls_metrics_mil.csv`
- `cls_report_mil.csv`
- `cls_metrics_mil.png`
- `attention_analysis.png` (if viz=True)
- `top_attention_cells.csv` (if viz=True)

#### C. `__train_vote()` - Majority Vote
**Location**: `models/classify.py:651`

**Process**:
1. Calls `__train_cell()` to get cell-level predictions
2. Aggregates cell predictions per sample:
   - Majority vote for predicted class
   - Average prediction score
3. Evaluates at sample level

**Output files**:
- `cls_predictions_vote.csv`
- `cls_metrics_vote.csv`
- `cls_report_vote.csv`
- `cls_metrics_vote.png`

#### D. `__train_cell()` - Cell-Level Classification
**Location**: `models/classify.py:548`

**Process**:
1. Uses embeddings directly (no aggregation)
2. Trains classifier on cell-level data
3. Predicts labels for each cell
4. Evaluates cell-level performance

**Output files** (saved to `cell_level_pred/` subdirectory):
- `cell_pred_test.csv`
- `cell_pred_train.csv`
- `cls_predictions_cell.csv`
- `cls_metrics_cell.csv`
- `cls_report_cell.csv`
- `cls_metrics_cell.png`

### Step 5: Cross-Validation (`__train_cv()`)

**Location**: `models/classify.py:279`

**Process**:
1. Validates all CV folds (checks for missing classes)
2. For each fold:
   - Splits data into train/test
   - Calls training function (avg/mil/vote)
   - Collects predictions and metrics
3. Aggregates results across folds:
   - Concatenates predictions
   - Concatenates metrics
4. Generates boxplots showing metric distributions across folds

**Output files** (saved to `cv/` subdirectory):
- `{prefix}_cv_predictions.csv` (all folds combined)
- `{prefix}_cv_metrics.csv` (all folds combined)
- `{prefix}_cv_predictions_train.csv`
- `{prefix}_cv_metrics_train.csv`
- `{prefix}_cv_metrics_boxplot.png` (metric distribution across folds)
- `{prefix}_cv_metrics_boxplot_train.png`

---

## 3. Directory Structure for Saved Results

### Base Structure
```
{experiment_save_dir}/
├── metrics/
│   └── classification/          # All classification metrics
│       ├── cls_predictions_avg_expr.csv
│       ├── cls_metrics_avg_expr.csv
│       ├── cls_report_avg_expr.csv
│       ├── cls_predictions_mil.csv
│       ├── cls_metrics_mil.csv
│       ├── cls_predictions_vote.csv
│       ├── cls_metrics_vote.csv
│       ├── cv/                 # Cross-validation results
│       │   ├── avg_cv_predictions.csv
│       │   ├── avg_cv_metrics.csv
│       │   ├── mil_cv_predictions.csv
│       │   ├── mil_cv_metrics.csv
│       │   ├── vote_cv_predictions.csv
│       │   ├── vote_cv_metrics.csv
│       │   ├── avg_cv_metrics_boxplot.png
│       │   └── ...
│       └── cell_level_pred/    # Cell-level results (if cls_level='cell')
│           ├── cell_pred_test.csv
│           ├── cell_pred_train.csv
│           ├── cls_predictions_cell.csv
│           └── cls_metrics_cell.csv
│
└── plots/
    └── evaluations/            # All classification plots
        ├── cls_metrics_avg_expr.png
        ├── cls_metrics_mil.png
        ├── cls_metrics_vote.png
        ├── attention_analysis.png  # MIL attention visualization
        └── cv/
            ├── avg_cv_metrics_boxplot.png
            ├── mil_cv_metrics_boxplot.png
            └── vote_cv_metrics_boxplot.png
```

### File Naming Conventions

**Single Split Results**:
- `cls_predictions_{method}.csv`: Predictions (label, pred, pred_score)
- `cls_metrics_{method}.csv`: Metrics (AUC, AUPRC, F1, Accuracy, Precision, Recall)
- `cls_report_{method}.csv`: Detailed classification report
- `cls_metrics_{method}.png`: Visualization plots

**CV Results**:
- `{method}_cv_predictions.csv`: All fold predictions combined
- `{method}_cv_metrics.csv`: All fold metrics combined (one row per fold)
- `{method}_cv_metrics_boxplot.png`: Boxplot of metrics across folds

**Train Set Results** (if evaluated):
- `cls_predictions_{method}_train.csv`
- `cls_metrics_{method}_train.csv`
- `cls_metrics_{method}_train.png`

---

## 4. Metrics Computed

For each evaluation, the system computes:

- **AUC** (Area Under ROC Curve)
- **AUPRC** (Average Precision)
- **F1 Score** (macro-averaged)
- **Accuracy**
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
- **Classification Report**: Per-class precision, recall, F1, support

---

## 5. Example Workflow

### Scenario: Patient-level classification with CV

1. **Config**:
   ```yaml
   train_funcs: [avg, mil, vote]
   cv: True
   onesplit: True
   cls_level: patient
   ```

2. **Execution**:
   - Loads train/test split and CV splits from JSON
   - For each training function (`avg`, `mil`, `vote`):
     - **Single Split**: Trains on train set, evaluates on test set
     - **CV**: Trains on 5 folds, evaluates on each fold's test set

3. **Outputs**:
   - **Single Split**: 3 methods × 2 sets (train/test) = 6 result sets
   - **CV**: 3 methods × 5 folds = 15 result sets, plus aggregated CV summaries

4. **Total Files Generated**:
   - 18 prediction CSVs (3 methods × 2 sets × 3 types: predictions, metrics, reports)
   - 6 plot PNGs (3 methods × 2 sets)
   - 15 CV prediction CSVs (3 methods × 5 folds)
   - 3 CV boxplots (one per method)
   - **Total: ~42 files**

---

## 6. Integration with Summary Generator

The `ResultsSummarizer` scans the `metrics/classification/` directory and collects:
- Single split metrics from `cls_metrics_*.csv`
- CV metrics from `cv/*_cv_metrics.csv` (aggregated or per-fold)

These are included in the comprehensive comparison tables with columns like:
- `classification_avg_expr_AUC`
- `classification_mil_AUC`
- `classification_vote_AUC`
- `classification_avg_expr_cv_AUC` (if CV results are aggregated)

---

## 7. Key Design Decisions

1. **Multiple Training Strategies**: Allows comparing different aggregation methods
2. **CV Support**: Provides robust performance estimates
3. **Separate Train/Test Evaluation**: Helps detect overfitting
4. **Organized Directory Structure**: Makes it easy to find specific results
5. **Flexible Configuration**: Supports different classification levels and models

---

## 8. Common Use Cases

### Binary Classification (Treatment Response)
```yaml
label_map:
  treatment_naive: 0
  neoadjuvant_chemo: 1
train_funcs: [avg, mil, vote]
cv: True
```

### Multiclass Classification (Cell Types)
```yaml
label_map:
  T_cell: 0
  B_cell: 1
  NK_cell: 2
train_funcs: [avg]
cv: False
onesplit: True
```

### Cell-Level Classification
```yaml
cls_level: cell
label_map:
  healthy: 0
  disease: 1
```
