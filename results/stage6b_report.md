# Stage 6B — Vanilla DAG-GNN Baseline: Final Report

> **Date:** August 19, 2026  
> **Status:** Complete

## Summary

A vanilla GraphSAGE-based GNN was trained on raw graph structure and node features (load_ff, x, y only) to predict critical-path delay mean and standard deviation. The model was evaluated on the Stage 6A dataset with 1397/296/307 train/val/test split, and compared against both a trivial baseline and the analytical SSTA baseline computed during Stage 6A.

**Key finding:** The vanilla GNN outperforms analytical SSTA on both mean (0.95× MAE) and std (0.37× MAE) predictions, with stable 3-seed performance and no overfitting.

---

## Dataset

| Property | Value |
|----------|-------|
| Total graphs | 2,000 |
| Train / Val / Test | 1,397 / 296 / 307 |
| Gates per graph | 6–14 (mean 9.97) |
| Reconvergence points | 2–8 (mean 2.85) |
| Feature normalization | Train-set mean/std only |
| Target normalization | Train-set mean/std only |

**Input features (per node):**
- `load_ff` — gate load capacitance
- `x`, `y` — spatial coordinates

**Targets (per graph):**
- `mc_labels.mean` — critical-path delay mean
- `mc_labels.std` — critical-path delay standard deviation

---

## Model Architecture

| Component | Specification |
|-----------|---------------|
| Message passing | 3 × GraphSAGE layers |
| Hidden dimension | 64 |
| Dropout | 0.15 |
| Readout | Mean pooling |
| Output head | 2-layer MLP (64 → 64 → 2) |
| Total parameters | 29,698 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE (sum over both outputs) |
| Early stopping | Patience=20 on validation loss |

**Design choices:**
- Directed edges following successors (source→sink), respecting DAG structure
- No physics-derived features (reserved for Stage 6C)
- No undirected message passing (vanilla baseline)

---

## Training

| Seed | Best Epoch | Best Val Loss | Train Time |
|------|-----------|---------------|------------|
| 42 | 91 | 0.0594 | 73.7s |
| 123 | 92 | 0.0548 | 70.3s |
| 999 | 108 | 0.0555 | 91.1s |

**Overfitting check:** Train/val loss ratios are 0.73–0.77, indicating healthy generalization without excessive capacity.

---

## Test Set Results (3-Seed Stability)

| Metric | Mean ± Std |
|--------|-----------|
| Mean delay MAE | **0.687 ± 0.004** |
| Mean delay relative error | **4.04% ± 0.03%** |
| Std delay MAE | **0.0337 ± 0.0002** |
| Std delay relative error | **4.67% ± 0.06%** |
| Avg inference time | **2.5 ms** |

**Stability:** All three seeds converged to very similar test errors (σ < 1% of mean), confirming the baseline is robust to initialization.

---

## Baseline Comparisons

| Baseline | Mean MAE | Mean Relative | Std MAE | Std Relative |
|----------|----------|---------------|---------|--------------|
| **GNN (ours)** | **0.687** | **4.04%** | **0.0337** | **4.67%** |
| Analytical SSTA | 0.722 | 4.57% | 0.0913 | 12.06% |
| Trivial (train mean) | 3.391 | 22.86% | 0.1233 | 18.33% |

**GNN vs Analytical SSTA:**
- Mean MAE ratio: 0.95× (GNN is 5% better)
- Std MAE ratio: 0.37× (GNN is 63% better)
- GNN beats analytical on both metrics across all 3 seeds

**GNN vs Trivial:**
- Mean MAE improvement: 79.7% relative
- Std MAE improvement: 72.6% relative

---

## Error Breakdown by Reconvergence Count

| nrecon | Count | Mean MAE | Mean Rel | Std MAE | Std Rel |
|--------|-------|----------|----------|---------|---------|
| 2 | 97 | 0.357–0.381 | 2.84–2.98% | 0.027–0.028 | 4.56–4.78% |
| 3 | 78 | 0.670–0.746 | 4.11–4.59% | 0.034–0.035 | 4.75–4.93% |
| 4 | 89 | 0.728–0.781 | 3.96–4.15% | 0.033–0.035 | 4.19–4.32% |
| 5 | 30 | 1.249–1.282 | 6.18–6.34% | 0.048–0.049 | 5.42–5.59% |
| 6 | 10 | 1.059–1.410 | 5.07–6.61% | 0.036–0.047 | 3.97–5.13% |
| 7 | 2 | 0.412–0.607 | 2.73–3.87% | 0.029–0.041 | 4.23–4.95% |
| 8 | 1 | 0.591–1.070 | 2.45–4.43% | 0.006–0.005 | 0.32–0.51% |

**Observation:** Error increases with topology complexity up to nrecon=5–6, then becomes noisy due to small sample sizes. The GNN generalizes to complex topologies but with degraded precision on the hardest cases — exactly the regime where Stage 6C's physics-informed features should help.

---

## Sanity Checks

### Overfitting
- Train/val loss ratios: 0.71–0.77 across seeds
- No divergence between train and validation curves
- Dropout (0.15) and early stopping (patience=20) are effective

### Trivial Baseline
- Predicting training-set mean gives 22.86% relative error on mean, 18.33% on std
- GNN reduces this to 4.04% and 4.67% respectively — a 5× improvement
- Confirms the GNN is learning genuine structure, not just dataset statistics

### Runtime
- Average inference time: 2.5 ms per graph
- Training time: ~70–90s per seed on CPU
- Well within budget for ablation studies

---

## Conclusions

1. **Vanilla GNN is a competent baseline:** 4.04% mean relative error, stable across 3 seeds, beats analytical SSTA by 5% on mean and 63% on std.

2. **No strawman:** The baseline is genuinely good-faith — it beats both trivial and analytical baselines, shows no overfitting, and generalizes across the stratified test set.

3. **Ready for Stage 6C comparison:** All metrics, checkpoints, and splits are frozen. The 0.37× std MAE ratio vs analytical SSTA is the number Stage 6C must beat or match.

4. **Limitation noted:** Error degrades on high-nrecon graphs (nrecon≥5), suggesting physics-informed features (Stage 6C) could particularly help with complex multi-reconvergence topologies.

---

## Files

| File | Description |
|------|-------------|
| `Vanilla DAG-GNN Baseline/dataset.py` | GraphDataset class, normalization, DataLoader creation |
| `Vanilla DAG-GNN Baseline/model.py` | VanillaDAGGNNSage architecture |
| `Vanilla DAG-GNN Baseline/train.py` | Training loop with early stopping |
| `Vanilla DAG-GNN Baseline/eval.py` | Evaluation metrics, nrecon breakdown, analytical comparison |
| `Vanilla DAG-GNN Baseline/run_stage6b.py` | Main script (3-seed training + evaluation) |
| `Vanilla DAG-GNN Baseline/results/vanilla_dag_gnn_results.json` | Full results (metrics, history, comparisons) |
| `Vanilla DAG-GNN Baseline/checkpoints/` | Best model checkpoints per seed |
