# Postmortem: Stage 6C Physics-Informed DAG-GNN — Honest Assessment

> **Date:** August 20, 2026  
> **Severity:** Informational — ablation completed, honest caveats documented  
> **Status:** Complete

## Issues Summary

Stage 6C completed a 3-way ablation (Vanilla → Tier A → Tier A+B) with identical training protocol. The results reveal important nuances about what "physics-informed" actually means in this context.

---

## Finding #1 — Tier A (Node-Level Sensitivities) Null Result Explained by Redundancy

### What Happened

Tier A added per-gate vth, l, w sensitivities to node features (3-dim → 6-dim). After 3-seed training:

| Metric | Vanilla | Tier A | Change | Bootstrap CI | Significant? |
|--------|---------|--------|--------|--------------|--------------|
| Mean MAE | 0.687 ± 0.004 | 0.685 ± 0.005 | -0.3% | [-0.025, +0.020] | No |
| Std MAE | 0.0337 ± 0.0002 | 0.0341 ± 0.0009 | +1.2% | — | No |

### Root Cause

Mechanism diagnostics reveal that ∂d/∂Vth is **perfectly correlated (r=1.00)** with the existing load_ff feature. This means Tier A adds no new information — the GNN already has access to the same signal through load_ff. The null result is due to **redundancy**, not "insufficiency."

### Honest Assessment

Tier A is a clean negative result with a clean mechanism explanation. The node-level physics features don't help because they're redundant with existing features. This is scientifically valuable — it tells us that future physics-informed features must provide information not already captured by geometry (load_ff, x, y).

---

## Finding #2 — Tier A+B Shows Large Improvement, GNN Contributes Meaningfully

### What Happened

Tier A+B added graph-level analytical SSTA features (sink_mean, sink_std) concatenated to the pooled embedding. Results:

| Metric | Vanilla | Tier A+B | Change | Bootstrap CI | Significant? |
|--------|---------|----------|--------|--------------|--------------|
| Mean MAE | 0.687 ± 0.004 | 0.542 ± 0.008 | **-21.1%** | [-0.192, -0.100] | **Yes** |
| Std MAE | 0.0337 ± 0.0002 | 0.0321 ± 0.0003 | **-4.7%** | — | **Yes** |

No-GNN residual baseline (MLP on sink_mean, sink_std, n_gates): mean MAE 0.9425 — far worse than Tier A+B's 0.5423. This proves the GNN contributes meaningfully beyond "residual correction of analytical."

### Root Cause

The analytical_ssta.sink_mean feature is 97.7% correlated with the MC mean label, so the GNN is learning to correct the analytical estimate. But the no-GNN baseline shows this correction requires graph structure — a plain MLP can't do it.

### Honest Assessment

Tier B's improvement is real and statistically significant. The GNN is doing something non-trivial. But the claim needs qualification: it's "GNN corrects analytical SSTA using graph structure," not "GNN learns physics from node features."

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

Analytical SSTA's linearization error compounds with more reconvergence points. On simple topologies (nrecon=2), analytical SSTA is already accurate. On complex topologies (nrecon≥5), analytical error is larger, and the GNN has more correction to learn.

### Honest Assessment

This is directionally consistent with "GNN corrects analytical error where it compounds," but the nrecon-n_gates confound and small-n buckets (n=10 at nrecon=6) prevent a strong causal claim.

---

## Finding #4 — Lockstep Verification Passed

The 6C vanilla run matches the 6B baseline exactly:
- Seed 42: 0.6829330921173096 (both)
- Seed 123: 0.6913238763809204 (both)
- Seed 999: 0.6875998973846436 (both)

This confirms the frozen baseline is untouched and the comparison is apples-to-apples.

---

## Finding #5 — Statistical Method Upgraded

Replaced the invalid z-test (sqrt((s₁²+s₂²)/3) with n=3) with paired per-graph bootstrap CI:
- 921 paired observations (307 graphs × 3 seeds)
- 10,000 bootstrap resamples
- Tier A: CI [-0.025, +0.020] includes 0 → not significant
- Tier A+B: CI [-0.192, -0.100] excludes 0 → significant

---

## Finding #6 — nrecon Stratification Reconciled

Earlier Stage 6A documentation referenced "boosting nrecon≥3 test coverage to 19 graphs." Actual test set: 210 graphs with nrecon≥3. The "19" was stale from a pre-fix version. Current `splits.json` uses `len(reconvergence_points)` consistently across all stages.

---

## Lessons Learned

1. **Redundancy is as important as informativeness**: Tier A's null result is explained by perfect correlation with load_ff, not by "insufficient signal." Future feature engineering must check for redundancy with existing features.
2. **No-GNN baseline is essential**: Without it, we couldn't distinguish "GNN corrects analytical" from "GNN learns physics." The 0.9425 vs 0.5423 gap proves graph structure matters.
3. **Bootstrap CI over z-test**: With n=3 seeds, the z-test is indefensible. Paired per-graph bootstrap is more powerful and honest.
4. **Be honest about "physics-informed"**: Injecting analytical results is useful but is closer to "GNN corrects analytical" than "GNN learns physics." The correlation check is essential context.
5. **Lockstep verification prevents silent drift**: Exact per-seed match between 6B and 6C vanilla runs confirms the frozen baseline is untouched.

## Prevention

- Added Tier A redundancy check (correlation with existing features) as mandatory diagnostic
- Added no-GNN residual baseline as standard ablation component
- Replaced z-test with paired per-graph bootstrap CI
- Added lockstep verification (per-seed match) between frozen baseline and new runs
- Documented nrecon stratification reconciliation
- Established that future "physics-informed" claims must include redundancy analysis and no-GNN baseline
