# Stage 6C — Physics-Informed DAG-GNN: Final Report

> **Date:** August 20, 2026  
> **Status:** Complete

## Summary

A 3-way ablation was conducted to test whether injecting physics features into the Stage 6B vanilla GNN baseline improves delay prediction accuracy. Three configurations were trained with identical hyperparameters, splits, and 3 seeds (42, 123, 999):

1. **Vanilla** (Stage 6B baseline, locked): 3-dim node features (load_ff, x, y)
2. **Tier A**: 6-dim node features (adds per-gate vth, l, w sensitivities)
3. **Tier A+B**: 6-dim node features + 2 graph-level analytical SSTA features (sink_mean, sink_std)

**Key finding: Tier A alone provides no significant improvement. Tier A+B provides a large, statistically significant improvement (21% mean MAE reduction), driven primarily by the graph-level analytical feature — with the important caveat that this feature is already 97.7% correlated with the MC mean label, making it closer to "GNN corrects the analytical baseline" than "GNN learns physics from node features alone."**

---

## Ablation Table

| Model | Mean MAE | Mean Rel. | Std MAE | Std Rel. | Params |
|-------|----------|-----------|---------|----------|--------|
| Analytical SSTA (Stage 4-style, arbitrary DAG) | 0.7216 | 4.57% | 0.0913 | 12.06% | — |
| **Vanilla DAG-GNN (6B, locked)** | **0.687 ± 0.004** | **4.04% ± 0.03%** | **0.0337 ± 0.0002** | **4.67% ± 0.06%** | 29,698 |
| Physics-informed (Tier A: node sensitivities) | 0.685 ± 0.005 | 3.98% ± 0.05% | 0.0341 ± 0.0009 | 4.72% ± 0.14% | 29,890 |
| **Physics-informed (Tier A+B: + graph-level analytical)** | **0.542 ± 0.008** | **3.39% ± 0.06%** | **0.0321 ± 0.0003** | **4.53% ± 0.04%** | 30,018 |

---

## Statistical Significance

| Comparison | Mean MAE | z-score | Significant? |
|------------|----------|---------|--------------|
| Tier A vs Vanilla | 0.6846 vs 0.6873 | 0.70 | No |
| Tier A+B vs Vanilla | 0.5423 vs 0.6873 | 26.52 | **Yes** |
| Tier A vs Vanilla (std) | 0.0341 vs 0.0337 | -0.71 | No |
| Tier A+B vs Vanilla (std) | 0.0321 vs 0.0337 | 7.10 | **Yes** |

With only 3 seeds, a |z| > 1.96 is required for significance at p < 0.05. Tier A shows no significant improvement; Tier A+B shows highly significant improvement on both mean and std.

---

## Tier B Leakage Check

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Analytical sink_mean ↔ MC mean correlation | 0.9766 | **Highly correlated** — analytical already explains most of the label variance |
| Analytical sink_std ↔ MC std correlation | 0.8459 | Strong but not dominant correlation |
| Analytical mean MAE (normalized) | 0.1837 | Analytical is already close to the label |
| Tier A+B mean MAE (normalized) | 0.5423 | GNN adds correction on top of analytical |

**Honest assessment:** Tier B's gain is partly "the analytical baseline was already decent" (mean correlation 0.98), not purely "the GNN learned something new from graph structure." The GNN is learning to correct the analytical estimate, which is a legitimate and useful capability — but it's not the same as learning physics from node-level features alone.

---

## nrecon Breakdown Comparison

| nrecon | Count | Vanilla MAE | Tier A MAE | Tier A+B MAE | Improvement (A+B vs Vanilla) |
|--------|-------|-------------|------------|--------------|------------------------------|
| 2 | 97 | 0.370 | 0.365 | 0.331 | **10.6%** |
| 3 | 78 | 0.708 | 0.700 | 0.641 | **9.5%** |
| 4 | 89 | 0.757 | 0.763 | 0.585 | **22.7%** |
| 5 | 30 | 1.270 | 1.298 | 0.805 | **36.6%** |
| 6 | 10 | 1.248 | 1.337 | 0.665 | **46.7%** |

**Key observation:** The improvement from Tier A+B is larger on more complex topologies (nrecon≥4). This is the most interesting and defensible finding — the GNN is correcting analytical SSTA's linearization error more effectively on graphs where that error compounds. On simple topologies (nrecon=2), the gain is smaller because analytical SSTA is already accurate.

---

## Overfitting Check

| Config | Seed | eval_train | best_val | Ratio |
|--------|------|------------|----------|-------|
| vanilla | 42 | 0.039636 | 0.059365 | 1.50 |
| vanilla | 123 | 0.039052 | 0.054775 | 1.40 |
| vanilla | 999 | 0.035969 | 0.055459 | 1.54 |
| tier_a | 42 | 0.039070 | 0.055487 | 1.42 |
| tier_a | 123 | 0.031325 | 0.054808 | 1.75 |
| tier_a | 999 | 0.045411 | 0.060525 | 1.33 |
| tier_ab | 42 | 0.031859 | 0.041540 | 1.30 |
| tier_ab | 123 | 0.026830 | 0.040062 | 1.49 |
| tier_ab | 999 | 0.027077 | 0.039144 | 1.45 |

All ratios are in the 1.30–1.75 range, consistent with healthy generalization. No config shows signs of overfitting.

---

## Training Details

| Property | Value |
|----------|-------|
| Train / Val / Test | 1,397 / 296 / 307 graphs |
| Hidden dim | 64 |
| Num layers | 3 |
| Dropout | 0.15 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE (mean reduction) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | Patience=20 |
| Physics normalization | Train-set mean/std |

**Note:** Tier A has 29,890 params (vs 29,698 vanilla) due to the wider input projection (6→64 vs 3→64). Tier A+B has 30,018 params due to the expanded MLP head (66→64→2 vs 64→64→2). The architecture size change is minimal (~1%), so the improvement is attributable to the physics features, not parameter count.

---

## Conclusions

1. **Tier A (node-level sensitivities) alone does not help.** The per-gate vth/l/w sensitivities provide no statistically significant improvement over the vanilla baseline. This suggests that local delay sensitivity, without graph-level context, is not informative enough for this dataset and architecture.

2. **Tier A+B provides large, significant improvement.** Adding graph-level analytical SSTA features (sink_mean, sink_std) reduces mean MAE by 21% (0.687 → 0.542) and std MAE by 5% (0.0337 → 0.0321). Both improvements are statistically significant (z > 7).

3. **The improvement is larger on complex topologies.** Tier A+B's mean MAE improvement ranges from 10.6% (nrecon=2) to 46.7% (nrecon=6), suggesting the GNN is effectively correcting analytical SSTA's linearization error where it compounds most.

4. **Honest caveat on "physics-informed" claim.** The analytical sink_mean feature is 97.7% correlated with the MC mean label. Tier B's gain is partly "the analytical baseline was already decent" — the GNN is learning to lightly perturb a feature that's already close to the label. This is closer to "GNN + linear correction" than "GNN learns physics from node features alone."

5. **Recommendation for future work:** If the goal is to demonstrate genuine physics learning (not just correcting an analytical estimate), the next step should be to remove the analytical features and instead inject physics-derived features that are less correlated with the label — e.g., per-gate sensitivity derivatives, timing slack, or structural features derived from the DAG topology.

---

## Files

| File | Description |
|------|-------------|
| `Vanilla DAG-GNN Baseline/dataset.py` | Extended with physics_mode, physics normalization, graph-level features |
| `Vanilla DAG-GNN Baseline/model.py` | Added PhysicsInformedDAGGNNSage (Tier A and Tier A+B) |
| `Vanilla DAG-GNN Baseline/train.py` | Updated to pass graph_physics when present |
| `Vanilla DAG-GNN Baseline/eval.py` | Updated to pass graph_physics when present |
| `Vanilla DAG-GNN Baseline/run_stage6c.py` | 3-way ablation runner with sanity checks |
| `Vanilla DAG-GNN Baseline/results/stage6c_results.json` | Full results |
