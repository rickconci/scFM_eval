# Evaluation Strategy: When to Run Which Evaluations

## Problem Statement

Currently, there's a mismatch between:
- **Biological signal & batch effects evaluations**: Designed for **cell-level labels** (cell types)
- **Classification tasks**: Can be **cell-level** (cell type annotation) or **patient-level** (subtype, chemo, outcome, etc.)

This leads to confusion where:
- A YAML filters to `timepoint=Pre` with `label_key: cell_types` (good for biological signal/batch effects)
- But then tries to classify `treatment_naive` vs `neoadjuvant_chemo` (patient-level, different labels)

## Solution: Context-Aware Evaluation Enabling

### Strategy: Keep YAMLs Per Task + Smart Defaults

**Keep the current structure** (one YAML per task) because:
1. ✅ **Clear intent**: Each YAML represents a specific task
2. ✅ **Explicit filters**: Easy to see what data is being used
3. ✅ **Less error-prone**: No complex cell subset logic per evaluation
4. ✅ **Maintainable**: Easy to understand and modify

**Add automatic evaluation enabling** based on context:
1. Detect what evaluations make sense given the data context
2. Provide warnings if evaluations don't match the context
3. Allow explicit override via `skip: True/False`

---

## Evaluation Context Rules

### Rule 1: Biological Signal & Batch Effects

**Should run when**:
- `label_key` is cell-level (e.g., `cell_types`, `cell_type`, `Cell types level 2`)
- OR classification is cell-level (`cls_level: cell`)
- AND we have multiple cell types (not binary patient-level labels)

**Should NOT run when**:
- Only patient-level labels (e.g., `Cancer_type`, `cohort`, `outcome`, `pre_post`)
- AND classification is patient-level
- AND no cell type information available

### Rule 2: Classification

**Always configurable** - can be cell-level or patient-level based on task.

---

## Implementation: Automatic Evaluation Detection

### Detection Logic

```python
def should_run_biological_signal_batch_effects(data_config, classification_config):
    """
    Determine if biological signal and batch effects evaluations make sense.
    
    Returns:
        bool: True if these evaluations are appropriate for the data context
    """
    label_key = data_config.get('label_key', '')
    
    # Check if label_key suggests cell-level labels
    cell_type_keywords = ['cell_type', 'cell_types', 'Cell types', 'cellType']
    is_cell_level_label = any(kw in label_key.lower() for kw in cell_type_keywords)
    
    # Check classification level
    cls_level = classification_config.get('params', {}).get('cls_level', 'patient')
    is_cell_level_classification = (cls_level == 'cell')
    
    # Check if we have multiple distinct labels (not just binary patient outcomes)
    # This would require checking the actual data, but we can infer from label_map
    label_map = classification_config.get('params', {}).get('label_map', {})
    has_multiple_labels = len(label_map) > 2
    
    # Biological signal/batch effects make sense if:
    # 1. We have cell-level labels, OR
    # 2. Classification is cell-level, OR  
    # 3. We have multiple cell types (inferred from label_map size)
    return is_cell_level_label or is_cell_level_classification or has_multiple_labels
```

---

## YAML Structure Recommendations

### For Cell Type Tasks (`cell_type/`, `all_cells/`)

```yaml
dataset:
  label_key: cell_types  # Cell-level labels
  filter:
    - column: timepoint
      values: [Pre]

evaluations:
  # ✅ These make sense - we have cell types
  - type: biological_signal
    skip: False  # Explicitly enable
    params:
      label_key: cell_types
  
  - type: batch_effects
    skip: False  # Explicitly enable
    params:
      label_key: cell_types
  
  # ✅ Cell-level classification
  - type: classification
    skip: False
    params:
      cls_level: cell
      label_map:
        T_cell: 0
        Cancer_cell: 1
        # ... more cell types
```

### For Patient-Level Tasks (`subtype/`, `chemo/`, `outcome/`, `pre_post/`)

```yaml
dataset:
  label_key: Cancer_type  # Patient-level labels
  filter:
    - column: timepoint
      values: [Pre]
    - column: Cancer_type
      values: [ER+, TNBC]

evaluations:
  # ❌ These DON'T make sense - no cell types available
  - type: biological_signal
    skip: True  # Explicitly disable (or auto-detect and skip)
  
  - type: batch_effects
    skip: True  # Explicitly disable (or auto-detect and skip)
  
  # ✅ Patient-level classification
  - type: classification
    skip: False
    params:
      cls_level: patient
      label_map:
        ER+: 0
        TNBC: 1
```

### For Mixed Tasks (e.g., `chemo/` with cancer cells only)

```yaml
dataset:
  label_key: cohort  # Patient-level labels
  filter:
    - column: timepoint
      values: [Pre]
    - column: cell_types
      values: [Cancer_cell]  # Filtered to one cell type

evaluations:
  # ❌ These DON'T make sense - only one cell type after filtering
  - type: biological_signal
    skip: True
  
  - type: batch_effects
    skip: True  # Could still run, but less meaningful with one cell type
  
  # ✅ Patient-level classification
  - type: classification
    skip: False
    params:
      cls_level: patient
      label_map:
        treatment_naive: 0
        neoadjuvant_chemo: 1
```

---

## Proposed Implementation

### Option A: Explicit Configuration (Recommended)

**Keep YAMLs explicit** - user specifies `skip: True/False` for each evaluation.

**Add validation/warnings** if configuration doesn't make sense:

```python
def validate_evaluation_config(data_config, evaluations_config):
    """
    Validate evaluation configuration and warn about mismatches.
    """
    label_key = data_config.get('label_key', '')
    
    for eval_config in evaluations_config:
        eval_type = eval_config.get('type')
        skip = eval_config.get('skip', False)
        
        if eval_type in ['biological_signal', 'batch_effects']:
            # Check if label_key suggests cell types
            is_cell_type_label = 'cell_type' in label_key.lower()
            
            if not skip and not is_cell_type_label:
                logger.warning(
                    f"{eval_type} evaluation is enabled but label_key='{label_key}' "
                    f"doesn't appear to be cell-level. This evaluation is designed for "
                    f"cell type preservation. Consider setting skip: True if this is "
                    f"a patient-level task."
                )
```

### Option B: Auto-Detection with Override

**Auto-detect** which evaluations make sense, but allow explicit override:

```python
def auto_enable_evaluations(data_config, evaluations_config, classification_config):
    """
    Automatically enable/disable evaluations based on context.
    User can still override with explicit skip: True/False.
    """
    for eval_config in evaluations_config:
        eval_type = eval_config.get('type')
        
        # If skip is explicitly set, respect it
        if 'skip' in eval_config:
            continue
        
        # Auto-detect based on context
        if eval_type in ['biological_signal', 'batch_effects']:
            should_run = should_run_biological_signal_batch_effects(
                data_config, classification_config
            )
            eval_config['skip'] = not should_run
            
            if not should_run:
                logger.info(
                    f"Auto-disabling {eval_type} evaluation: not appropriate for "
                    f"patient-level task (label_key='{data_config.get('label_key')}')"
                )
```

---

## Recommendation

**Use Option A (Explicit Configuration)** with validation warnings:

1. ✅ **Keep YAMLs explicit** - user controls what runs
2. ✅ **Add validation** - warn if configuration doesn't make sense
3. ✅ **Clear documentation** - document which evaluations make sense for which tasks
4. ✅ **Simple and maintainable** - no complex auto-detection logic

**Benefits**:
- Clear intent in YAML files
- User has full control
- Warnings catch mistakes
- Easy to understand and maintain

**Example YAML with proper configuration**:

```yaml
# brca_full/all_cells/scimilarity.yaml
dataset:
  label_key: cell_types  # Cell-level - good for biological signal/batch effects
  filter:
    - column: timepoint
      values: [Pre]

evaluations:
  # ✅ These make sense - we have cell types
  - type: biological_signal
    skip: False
    params:
      label_key: cell_types
  
  - type: batch_effects
    skip: False
    params:
      label_key: cell_types
  
  # ❌ Classification doesn't make sense here - no classification task defined
  # (or if it does, it should be cell-level)
  - type: classification
    skip: True  # No classification task for "all_cells"
```

```yaml
# brca_full/chemo/scimilarity.yaml
dataset:
  label_key: cohort  # Patient-level - NOT good for biological signal/batch effects
  filter:
    - column: timepoint
      values: [Pre]
    - column: cell_types
      values: [Cancer_cell]

evaluations:
  # ❌ These DON'T make sense - no cell types (filtered to one type)
  - type: biological_signal
    skip: True
  
  - type: batch_effects
    skip: True
  
  # ✅ Patient-level classification
  - type: classification
    skip: False
    params:
      cls_level: patient
      label_map:
        treatment_naive: 0
        neoadjuvant_chemo: 1
```

---

## Summary

**Keep the current YAML structure** (one per task) and:
1. **Make evaluations explicit** - user sets `skip: True/False`
2. **Add validation warnings** - alert if configuration doesn't match context
3. **Document best practices** - which evaluations for which tasks
4. **Keep it simple** - no complex auto-detection that might be wrong

This approach is:
- ✅ Clear and explicit
- ✅ Easy to understand
- ✅ Maintainable
- ✅ Catches mistakes with warnings
- ✅ Gives user full control
