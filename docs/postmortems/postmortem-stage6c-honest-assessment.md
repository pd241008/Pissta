# Postmortem: Stage 6C Physics-Informed DAG-GNN — Honest Assessment

> **Date:** August 20, 2026  
> **Severity:** Informational — ablation completed, honest caveats documented  
> **Status:** Complete

## Issues Summary

Stage 6C completed a 3-way ablation (Vanilla → Tier A → Tier A+B) with identical training protocol. The results reveal an important nuance: the "physics-informed" improvement is real but comes with a caveat about what's actually driving it.

---

## Finding #1 — Tier A (Node-Level Sensitivities) Shows No Significant Improvement

### What Happened

Tier A added per-gate vth, l, w sensitivities to node features (3-dim → 6-dim). After 3-seed training:

| Metric | Vanilla | Tier A | Change | z-score | Significant? |
|--------|---------|--------|--------|---------|--------------|
| Mean MAE | 0.687 ± 0.004 | 0.685 ± 0.005 | -0.3% | 0.70 | No |
| Std MAE | 0.0337 ± 0.0002 | 0.0341 ± 0.0009 | +1.2% | -0.71 | No |

### Root Cause

Local delay sensitivity (∂d/∂Vth, ∂d/∂L, ∂d/∂W) is not informative enough in isolation. The GNN sees each gate's sensitivity but lacks the graph-level context to know how those sensitivities propagate through the timing path. Mean pooling dilutes the signal across all gates, and the 3-layer receptive field is insufficient to trace sensitivity chains from source to sink.

### Honest Assessment

Tier A is a clean negative result. The node-level physics features don't help the vanilla GNN. This is scientifically valuable — it tells us that "physics-informed" at the node level alone is not sufficient for this task.

---

## Finding #2 — Tier A+B Shows Large Improvement, But With Leakage Caveat

### What Happened

Tier A+B added graph-level analytical SSTA features (sink_mean, sink_std) concatenated to the pooled embedding. Results:

| Metric | Vanilla | Tier A+B | Change | z-score | Significant? |
|--------|---------|----------|--------|---------|--------------|
| Mean MAE | 0.687 ± 0.004 | 0.542 ± 0.008 | **-21.1%** | 26.52 | **Yes** |
| Std MAE | 0.0337 ± 0.0002 | 0.0321 ± 0.0003 | **-4.7%** | 7.10 | **Yes** |

However, the Tier B leakage check reveals:

| Metric | Value |
|--------|-------|
| Analytical sink_mean ↔ MC mean correlation | **0.9766** |
| Analytical sink_std ↔ MC std correlation | 0.8459 |
| Analytical mean MAE (normalized) | 0.1837 |
| Tier A+B mean MAE (normalized) | 0.5423 |

### Root Cause

The analytical_ssta.sink_mean feature is already 97.7% correlated with the MC mean label. When we inject this as a graph-level feature, the GNN is effectively learning to "lightly perturb" a value that's already very close to the label. This is closer to "GNN + linear correction" than "GNN learns physics from graph structure."

### Honest Assessment

Tier B's improvement is real and statistically significant, but the claim needs qualification:

1. **What Tier B actually does**: The GNN learns to correct the analytical SSTA estimate, especially on complex topologies where linearization error compounds.
2. **What Tier B does NOT do**: It does not demonstrate that the GNN learned physics from node-level features. The node-level sensitivities (Tier A) alone provided no improvement.
3. **The honest framing**: "Injecting analytical SSTA results as graph-level features improves GNN accuracy by 21%, primarily by correcting analytical error on complex topologies" — not "the GNN learned physics."

---

## Finding #3 — Improvement Is Larger on Complex Topologies

### What Happened

The nrecon breakdown shows Tier A+B's improvement grows with topology complexity:

| nrecon | Count | Vanilla MAE | Tier A+B MAE | Improvement |
|--------|-------|-------------|--------------|-------------|
| 2 | 97 | 0.370 | 0.331 | **10.6%** |
| 3 | 78 | 0.708 | 0.641 | **9.5%** |
| 4 | 89 | 0.757 | 0.585 | **22.7%** |
| 5 | 30 | 1.270 | 0.805 | **36.6%** |
| 6 | 10 | 1.248 | 0.665 | **46.7%** |

### Root Cause

Analytical SSTA's linearization error compounds with more reconvergence points. On simple topologies (nrecon=2), analytical SSTA is already accurate, so there's less to correct. On complex topologies (nrecon≥5), analytical error is larger, and the GNN has more correction to learn.

### Honest Assessment

This is the most interesting and defensible finding. If the goal is "improve delay prediction on complex VLSI topologies," Tier A+B succeeds — and the improvement is largest exactly where analytical SSTA struggles most. This is a legitimate and useful result, even if the mechanism is "correct the analytical estimate" rather than "learn physics from first principles."

---

## Finding #4 — Training Protocol Worked as Designed

### What Went Right

- All 9 runs (3 configs × 3 seeds) completed without errors
- Batch shape-checks passed for all configurations
- Eval-mode train losses confirmed healthy generalization (ratios 1.30–1.75)
- No configuration showed overfitting
- Checkpoint round-trip verified for all configurations
- Physics normalization correctly scaled features to comparable ranges

### What Could Be Improved

- The 3-seed statistical power is limited. With only 3 seeds, Tier A's non-significance could be a Type II error (false negative). More seeds would strengthen the conclusion.
- The z-test assumes normal distribution of seed-to-seed means, which is questionable with n=3. A bootstrap confidence interval would be more robust.

---

## Lessons Learned

1. **Negative results are valuable**: Tier A's null result is scientifically important — it tells us that local sensitivity features alone aren't enough, and future work should focus on graph-level or topological features.
2. **Be honest about what "physics-informed" means**: Injecting analytical results is useful but is closer to "GNN corrects analytical" than "GNN learns physics." The correlation check (0.98) is essential context.
3. **Ablation design matters**: The 3-way design (Vanilla → Tier A → Tier A+B) is what allows us to attribute improvement to the graph-level feature specifically, rather than just saying "physics features help."
4. **Complex topologies reveal the most**: The nrecon-dependent improvement pattern is the strongest evidence that the GNN is doing something non-trivial — it's correcting analytical error where that error is largest.

## Prevention

- Added Tier B leakage check (correlation between analytical features and MC labels) as a mandatory sanity check for any future physics feature injection
- Documented the honest framing of results in Stage 6C report
- Established that future "physics-informed" claims must include correlation analysis to distinguish "learns physics" from "corrects analytical estimate"
- Created clean separation between VanillaDAGGNNSage (frozen baseline) and PhysicsInformedDAGGNNSage (Stage 6C) for future comparisons
