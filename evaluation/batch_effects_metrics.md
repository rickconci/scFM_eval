# Batch Effects Evaluation Metrics

This document explains the mathematics and theory behind each metric used to evaluate batch effect correction in single-cell RNA sequencing embeddings.

## Overview

Batch effects are systematic variations in gene expression that arise from technical factors (e.g., different sequencing runs, laboratories, or processing dates) rather than biological differences. Effective batch correction should:
1. **Remove batch effects**: Cells from different batches should be well-mixed in embedding space
2. **Preserve biological signal**: Cell types and biological states should remain well-separated

The metrics below quantify these two aspects of batch correction quality.

---

## 1. ASW_batch (Average Silhouette Width for Batch)

### Mathematical Definition

The silhouette coefficient measures how similar a cell is to its own batch compared to other batches. For cell $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ = average distance from cell $i$ to all other cells in the **same batch**
- $b(i)$ = average distance from cell $i$ to all other cells in the **nearest different batch**

The ASW_batch is the average silhouette coefficient across all cells:

$$\text{ASW\_batch} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

### Interpretation

- **Range**: $[-1, 1]$
- **Lower is better**: Values close to 0 or negative indicate good batch mixing (cells are as similar to other batches as to their own)
- **Higher values**: Indicate strong batch separation (bad batch correction)

### Why It Matters

ASW_batch directly measures whether batch structure is still present in the embedding. After batch correction, cells should not cluster by batch, so ASW_batch should be low.

---

## 2. ASW_label/batch (Silhouette Width Balancing Label and Batch)

### Mathematical Definition

This metric balances two competing goals:
1. **Label separation**: Cells of the same type should be close together
2. **Batch mixing**: Cells from different batches should be well-mixed

The metric computes a silhouette score that accounts for both factors:

$$\text{ASW\_label/batch} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \text{ASW}(\ell)$$

where $\text{ASW}(\ell)$ is the average silhouette width for label $\ell$, computed as:

$$\text{ASW}(\ell) = \frac{1}{n_\ell} \sum_{i \in \ell} \frac{b_\text{label}(i) - a_\text{batch}(i)}{\max(a_\text{batch}(i), b_\text{label}(i))}$$

where:
- $a_\text{batch}(i)$ = average distance from cell $i$ to cells in the **same batch** but **different label**
- $b_\text{label}(i)$ = average distance from cell $i$ to cells with the **same label** but **different batch**

### Interpretation

- **Range**: $[-1, 1]$
- **Higher is better**: Positive values indicate good label separation while maintaining batch mixing
- **Ideal value**: Close to 1, meaning cells of the same type are close together, but batch structure is removed

### Why It Matters

This is a key metric because it directly measures the trade-off between preserving biological signal (cell types) and removing technical noise (batch effects). Good batch correction should maximize this metric.

---

## 3. PCR_batch (Principal Component Regression on Batch)

### Mathematical Definition

PCR_batch measures how much of the variance in the embedding can be explained by batch labels. It works by:

1. **PCA on embeddings**: Compute principal components (PCs) of the embedding matrix $X \in \mathbb{R}^{n \times d}$:
   $$X = U \Sigma V^T$$
   where $U$ contains the principal components.

2. **Regression**: For each PC $u_k$, regress it against batch labels:
   $$u_k = \beta_0 + \beta_1 \cdot \text{batch} + \epsilon$$

3. **Variance explained**: Compute $R^2$ for each PC-batch regression:
   $$R^2_k = 1 - \frac{\text{SSE}_k}{\text{SST}_k}$$
   where SSE is sum of squared errors and SST is total sum of squares.

4. **Weighted average**: Weight by variance explained by each PC:
   $$\text{PCR\_batch} = \frac{\sum_{k=1}^{K} \lambda_k \cdot R^2_k}{\sum_{k=1}^{K} \lambda_k}$$
   where $\lambda_k$ is the eigenvalue (variance) of PC $k$, and $K$ is typically 50.

### Interpretation

- **Range**: $[0, 1]$
- **Lower is better**: Values close to 0 indicate that batch explains little variance in the embedding
- **Higher values**: Indicate that batch structure is still prominent in the embedding space

### Why It Matters

PCR_batch quantifies how much batch information "leaks" into the principal components of the embedding. After correction, batch should explain minimal variance in the embedding space.

---

## 4. iLISI (Integration Local Inverse Simpson's Index)

### Mathematical Definition

iLISI measures batch mixing at the local neighborhood level. For each cell $i$:

1. **Find k-nearest neighbors**: Identify the $k$ nearest neighbors of cell $i$ in embedding space
2. **Count batch diversity**: Compute the Local Inverse Simpson's Index:
   $$\text{LISI}_i = \frac{1}{\sum_{b=1}^{B} p_{i,b}^2}$$
   
   where $p_{i,b}$ is the proportion of neighbors of cell $i$ that belong to batch $b$, and $B$ is the number of batches.

3. **Normalize**: The iLISI score is normalized to $[0, 1]$:
   $$\text{iLISI} = \frac{\text{median}(\text{LISI}_i) - 1}{B - 1}$$

### Interpretation

- **Range**: $[0, 1]$
- **Higher is better**: Values close to 1 indicate perfect batch mixing (each cell's neighborhood contains cells from all batches equally)
- **Lower values**: Indicate that cells tend to have neighbors from the same batch

### Why It Matters

iLISI provides a local, neighborhood-based measure of batch mixing. Unlike global metrics, it captures whether batch effects are removed at fine-grained scales, which is important for downstream analyses like trajectory inference or clustering.

---

## 5. cLISI (Cell type Local Inverse Simpson's Index)

### Mathematical Definition

cLISI measures cell type separation (biological signal preservation). It uses the same LISI framework as iLISI but applied to cell type labels:

1. **Find k-nearest neighbors**: Identify the $k$ nearest neighbors of cell $i$
2. **Count label diversity**: Compute LISI for cell types:
   $$\text{LISI}_i = \frac{1}{\sum_{\ell=1}^{L} p_{i,\ell}^2}$$
   
   where $p_{i,\ell}$ is the proportion of neighbors of cell $i$ that belong to cell type $\ell$, and $L$ is the number of cell types.

3. **Normalize and invert**: Since we want **low** diversity (high separation), we invert:
   $$\text{cLISI} = \frac{L - \text{median}(\text{LISI}_i)}{L - 1}$$

### Interpretation

- **Range**: $[0, 1]$
- **Higher is better**: Values close to 1 indicate perfect cell type separation (each cell's neighborhood contains only cells of the same type)
- **Lower values**: Indicate poor cell type separation (neighborhoods are mixed across cell types)

### Why It Matters

cLISI ensures that batch correction doesn't over-correct and destroy biological signal. Good batch correction should have high cLISI (good separation) and high iLISI (good mixing), indicating that biological structure is preserved while technical structure is removed.

---

## 6. kBET (k-nearest neighbor Batch Effect Test)

### Mathematical Definition

kBET tests whether the batch distribution in each cell's neighborhood matches the global batch distribution (null hypothesis: batches are well-mixed).

For each cell $i$:

1. **Find k-nearest neighbors**: Identify $k$ nearest neighbors
2. **Count batches**: Count how many neighbors belong to each batch: $n_{i,b}$ for batch $b$
3. **Expected counts**: Under the null hypothesis (well-mixed), expected counts are:
   $$E_{i,b} = k \cdot p_b$$
   where $p_b$ is the global proportion of cells in batch $b$

4. **Chi-square test**: Compute chi-square statistic:
   $$\chi^2_i = \sum_{b=1}^{B} \frac{(n_{i,b} - E_{i,b})^2}{E_{i,b}}$$

5. **Rejection rate**: Count the proportion of cells where $\chi^2_i > \chi^2_{\alpha, B-1}$ (rejecting the null):
   $$\text{kBET} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\chi^2_i \leq \chi^2_{\alpha, B-1}]$$

### Interpretation

- **Range**: $[0, 1]$
- **Higher is better**: Values close to 1 indicate that most neighborhoods have batch distributions matching the global distribution (good mixing)
- **Lower values**: Indicate that neighborhoods are biased toward specific batches (poor mixing)

### Why It Matters

kBET provides a statistical test of batch mixing. It's particularly useful because it accounts for the expected batch distribution, making it robust to imbalanced batch sizes.

---

## 7. batch_effects_score (Summary Score)

### Mathematical Definition

The batch_effects_score is a composite metric that combines all individual metrics into a single score:

1. **Normalize metrics to [0, 1]**: For metrics where lower is better (ASW_batch, PCR_batch), invert:
   $$\text{normalized\_metric} = \frac{1}{1 + \text{original\_metric}}$$
   
   For metrics where higher is better (ASW_label/batch, iLISI, cLISI, kBET), use directly.

2. **Average**: Compute the mean of all normalized metrics:
   $$\text{batch\_effects\_score} = \frac{1}{M} \sum_{m=1}^{M} \text{normalized\_metric}_m$$
   
   where $M$ is the number of valid (non-NaN) metrics.

### Interpretation

- **Range**: $[0, 1]$
- **Higher is better**: Values close to 1 indicate excellent batch correction overall
- **Provides**: A single number to compare different batch correction methods

### Why It Matters

The summary score allows for quick comparison across methods and provides a single metric for optimization or ranking.

---

## Summary Table

| Metric | Range | Better | What It Measures |
|--------|-------|--------|------------------|
| **ASW_batch** | $[-1, 1]$ | Lower | Global batch separation |
| **ASW_label/batch** | $[-1, 1]$ | Higher | Balance of label separation and batch mixing |
| **PCR_batch** | $[0, 1]$ | Lower | Variance explained by batch in PCA space |
| **iLISI** | $[0, 1]$ | Higher | Local batch mixing (neighborhood level) |
| **cLISI** | $[0, 1]$ | Higher | Local cell type separation (biological signal) |
| **kBET** | $[0, 1]$ | Higher | Statistical test of batch mixing |
| **batch_effects_score** | $[0, 1]$ | Higher | Composite score combining all metrics |

---

## References

- **ASW metrics**: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of computational and applied mathematics*, 20, 53-65.

- **LISI metrics**: Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature methods*, 16(12), 1289-1296.

- **kBET**: Büttner, M., et al. (2019). A test metric for assessing single-cell RNA-seq batch correction. *Nature methods*, 16(1), 43-49.

- **PCR**: Büttner, M., et al. (2019). scIB: a Python package for benchmarking batch correction methods for single-cell RNA-seq data. *Bioinformatics*, 35(14), i41-i50.

- **scIB package**: Heumos, L., et al. (2023). Best practices for single-cell analysis across modalities. *Nature Reviews Genetics*, 24(8), 550-572.
