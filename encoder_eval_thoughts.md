# Omnicell encoder — why the bio score is weak on DKD

This note summarizes what the current debug pipeline actually does, how it matches
(or doesn’t match) how the encoder was trained, which knobs *can* change bio/batch
scores, and what I think is most likely driving the “cell types mix in the UMAP”
observation.

## TL;DR

- We have **three ways** to push cells through the encoder plus a **PCA baseline**:
  - **A. Singleton** sets `(B, 1, G)` — set-level `lat` is the per-cell embedding.
  - **B. Joint set** `(1, N, G)` + `return_cell_embeddings=True` — SetNorm sees *all*
    `N` cells together (i.e. a mixture of cell types + batches).
  - **C. Grouped-by-label** `(1, n_ct, G)` per cell type + `return_cell_embeddings=True`
    — SetNorm statistics are computed **within one cell type**, matching the
    perturbseq training regime (sets of same-cell-type cells).
- We now also compute **ASW_label / ASW_batch / iLISI / cLISI** (same definitions as
  `scFM_eval/evaluation/{eval,batch_effects}.py`) and save them to `metrics.csv`.
- Nothing is *obviously* wrong with the pipeline — the encoder is **fed count-like data**,
  mapped genes line up with the manager, and the joint path produces finite,
  non-degenerate embeddings. But there are two subtle issues that probably move the
  needle on bio score: **(1) input-set composition vs training** and **(2) the
  joint path mixing all cell types into one SetNorm statistic**. These are pipeline
  issues, not (necessarily) model issues. Method **C** is the test that disambiguates
  them.

## What the script now does

```text
load_encoder (EMA by default)
  ↓
load_dkd_encoder_data  → X_shared (N, 8369), cell_types, batches
  ↓
(1) singleton   (B,1,G)   return_cell_embeddings=False   → lat (N, L)
(2) joint_set   (1,N,G)   return_cell_embeddings=True    → cell_emb (N, L)
(3) grouped     (1,n_ct,G) per cell_type, concat in order → (N, L)
(4) PCA         sc.tl.pca on X_shared                    → (N, 50)
  ↓
UMAPs (Scanpy style) + metrics (ASW_label, ASW_batch, iLISI_med, cLISI_med)
```

The metrics script mirrors `scFM_eval`:

- `ASW_label` = `silhouette(emb, cell_type)` rescaled to `[0, 1]` (higher = better bio).
- `ASW_batch` = `1 − silhouette(emb, batch)` rescaled to `[0, 1]` (higher = better batch mixing).
- `iLISI_med` = median pure-python LISI over the neighbor graph with `batch` labels
  (higher = better batch mixing; raw range `[1, n_batches]`).
- `cLISI_med` = same with `label` labels (lower = better cell-type separation; raw range `[1, n_labels]`).

PCA is included as the reference that `scFM_eval` considers the canonical PRE state.

## Have we done anything glaringly wrong?

No single blocker, but here are the things that *could* materially hurt bio score,
ordered by how likely I think they are:

1. **Input-set composition does not match training.**
   - Training uses sets sampled from **(cell_type, pert)** partitions in perturbseq and
     **per-file** samples in sharded/cellxgene data — i.e. the model expects the set to
     be relatively homogeneous in cell type (especially in `obs`). At inference we
     currently pass **mixtures**:
     - Method **B** pushes *all 2000* cells through one SetNorm → the
       mean/var estimated inside the encoder is dominated by the frequency of cell types
       in DKD, not by a single type’s distribution.
     - Method **A** sets `S = 1`. Mathematically `SetNorm` over a single element
       normalizes each feature by its single value, which is fine but not the regime
       the weights were trained in.
   - Method **C** is the fair test: SetNorm sees **one cell type at a time**, matching
     training. If **C beats B and A on `ASW_label` / `cLISI`**, the earlier weak bio
     score was **a set-composition artefact**, not a model failure.

2. **“Train on count-like” vs what we send.**
   - The encoder is trained on count-like/float32 ints (cellxgene counts). We currently
     read `adata.raw.to_adata()` when `use_raw=True` (default) — if the DKD h5ad’s raw
     layer is log-normalized rather than counts, the encoder sees an out-of-distribution
     scale. The map-genes diagnostic prints `frac_int≈ …` on the shared slice; if that
     is far below 1.0 we’re sending normalized data.

3. **Gene resolution.**
   - `map_mode=feature_name` matches on symbols; DKD has duplicate / alias gene symbols
     that mis-map into the global 90k list (we saw 20 genes dropped by the mapper and a
     good fraction of Ensembl-like names). `feature_id` with `gene_id` / `feature_id`
     populating `var_names` is strictly better when available. Mismatched columns are
     **implicit zeros** at test time — even modest miss rates can degrade any single
     cell’s embedding.

4. **EMA vs raw weights.**
   - We default to `load_avg=True` (EMA). That’s usually safer, but the raw checkpoint
     can look quite different on downstream scores. `--load-raw-weights` toggles it.

5. **Checkpoint epoch.**
   - The checkpoint we use is `epoch_5`. If the encoder was trained jointly with the
     generator (no `freeze_encoder`), the earliest checkpoints can be bio-conservative
     and batch-conservative — worth sanity-checking a later epoch if available.

6. **Set size at inference.**
   - Training uses `experiment.set_size: 100`. Method **C** with chunk size
     `≤ max_set_size=4096` is much closer to this regime than Method **B** (single set
     of `N=2000`). If bio score with **C** improves monotonically as we lower
     `max_set_size` towards ~100, that’s strong evidence the encoder is sensitive to
     the set statistics it sees at inference.

## What I do *not* think is wrong

- **Dimensionality / in_dim.** `input_projection.in_features == len(shared_genes) == 8369`
  — that check passes in the current run.
- **Gene ordering.** `map_genes` produces `shared_gene_ids` that index into
  `adata_mapped.X`; we then slice exactly those columns. No hidden permutation.
- **Encoder output shape handling.** For `S=1` the set-level `lat` is the per-cell
  vector; for `(1,N,G)` we correctly take the second return (per-cell) with
  `return_cell_embeddings=True`. Both are the documented encoder API.
- **UMAP pipeline.** PCA → neighbors → UMAP, same defaults as
  `scFM_eval/viz/visualization.py` (`wspace=0.4`, `frameon=False`, `dpi=200`).

## What to look at first on the next run

1. Open `metrics.csv`. Compare **singleton / joint / grouped / pca** on
   `ASW_label`, `iLISI_med`, `cLISI_med`. In the “pipeline is wrong” scenario:
   - **PCA** beats singleton and joint on `ASW_label` / `cLISI`.
   - **Grouped** beats both singleton and joint; ideally also beats or matches PCA.
   In the “model is weak at batch-effect / bio separation” scenario:
   - Even grouped stays *below* PCA on `ASW_label` / `cLISI`.
2. In `print_dkd_map_diagnostics` output, confirm:
   - `var columns in [shared]` is high (we saw ~N/N% on the current run).
   - `dtype: integer=True` (or `frac_int ≈ 1.0`) on the shared slice — if not,
     the `--no-raw-layer` flag is needed because the h5ad’s main `X` is probably
     counts.
3. Re-run with `--load-raw-weights` to see if the raw (non-EMA) encoder changes
   things materially.
4. Optional follow-up (not in the script yet): add a `--max-set-size K` knob to
   method **C** to test sensitivity to chunk size (expect bio score to peak near
   the training `set_size ≈ 100`).

## Concrete takeaway

The most defensible test before blaming the model is Method **C** (per-cell-type
sets). If bio/cLISI improves substantially over the joint path and approaches PCA,
the earlier UMAPs that “mixed cell types” were mostly an artefact of **feeding a
set that is a mixture of cell types** into a SetNorm encoder trained on homogeneous
sets — *not* a failure of the encoder to separate types. If **C** still looks poor,
we likely have a checkpoint/weights or data-scale (counts vs lognorm) problem
upstream, in that order of probability.
