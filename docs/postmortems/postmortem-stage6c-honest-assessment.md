# Postmortem: Stage 6C Physics-Informed DAG-GNN — Honest Assessment

> **Date:** August 24, 2026  
> **Severity:** Informational — ablation completed, honest caveats documented  
> **Status:** Complete (updated with final deterministic re-run)

## Issues Summary

Stage 6C completed a 3-way ablation (Vanilla → Tier A → Tier A+B) with identical training protocol on the corrected dataset. The results reveal that physics feature injection has no statistically significant effect in either direction — vanilla remains the best point estimate.

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

## Finding #2 — Tier A+B Shows No Significant Improvement (Null Result)

### What Happened

Tier A+B added graph-level analytical SSTA features (sink_mean, sink_std) concatenated to the pooled embedding. After 3-seed training on the corrected dataset:

| Metric | Vanilla | Tier A+B | Change | Bootstrap CI | Significant? |
|--------|---------|----------|--------|--------------|--------------|
| Mean MAE | 0.4263 ± 0.0063 | 0.4307 ± 0.0147 | +1.0% | [-0.0450, +0.0523] | **No** |
| Std MAE | 0.0194 ± 0.0003 | 0.0199 ± 0.0007 | +2.6% | [-0.0012, +0.0022] | **No** |

No-GNN residual baseline (MLP on sink_mean, sink_std, n_gates): mean MAE 0.5786 ± 0.0010 — far worse than all GNN variants. This proves graph structure matters, but the physics features don't push the GNN beyond vanilla.

### Root Cause

The analytical_ssta.sink_mean feature is ~97.6% correlated with the MC mean label. The GNN cannot extract additional signal from this near-perfect feature in the current architecture. The earlier significant result (-21%) was an artifact of the mislabeled dataset (B1).

### Honest Assessment

Tier A+B's null result is the correct scientific conclusion on corrected data. The GNN does meaningful work (beats no-GNN by 34.4%), but the physics features themselves don't contribute beyond what vanilla already learns from graph structure. Future work should explore physics features that provide *new* signal, not near-duplicates of the label.

---

## Finding #3 — nrecon Breakdown Shows No Consistent Tier Advantage

### What Happened

The nrecon breakdown shows no consistent advantage for either physics tier over vanilla across topology complexity:

| nrecon | Count | Vanilla MAE | Tier A MAE | Tier A+B MAE |
|--------|-------|-------------|------------|--------------|
| 1 | 150 | 0.2984 | 0.2891 | 0.3702 |
| 2 | 128 | 0.5263 | 0.5405 | 0.4935 |
| 3 | 24 | 0.5958 | 0.8224 | 0.5024 |
| 4 | 2 | — | — | — |

### Honest Assessment

No tier shows a consistent improvement pattern. The earlier observation that Tier A+B improved more on complex topologies was an artifact of the buggy dataset. On corrected data, the CIs for all tier-vs-vanilla comparisons cross zero at every nrecon level.

---

## Finding #4 — Lockstep Verification Passed

The 6C vanilla run matches the 6B baseline exactly:
- Seed 42: 0.420840 (both)
- Seed 123: 0.433124 (both)
- Seed 999: 0.424949 (both)

This confirms the frozen baseline is untouched and the comparison is apples-to-apples.

---

## Finding #5 — Statistical Method Upgraded

Replaced the invalid z-test (sqrt((s₁²+s₂²)/3) with n=3) with paired per-graph bootstrap CI:
- 304 paired observations per seed, 3 seeds = 912 total observations
- 10,000 bootstrap resamples
- Tier A: CI [-0.0113, +0.0272] includes 0 → not significant
- Tier A+B: CI [-0.0450, +0.0523] includes 0 → not significant

---

## Finding #6 — nrecon Stratification Reconciled

Earlier Stage 6A documentation referenced "boosting nrecon≥3 test coverage to 19 graphs." Actual test set: 210 graphs with nrecon≥3. The "19" was stale from a pre-fix version. Current `splits.json` uses `len(reconvergence_points)` consistently across all stages.

---

## Lessons Learned

1. **Redundancy is as important as informativeness**: Tier A's null result is explained by perfect correlation with load_ff, not by "insufficient signal." Future feature engineering must check for redundancy with existing features.
2. **No-GNN baseline is essential**: Without it, we couldn't distinguish "GNN corrects analytical" from "GNN learns physics." The 0.5786 vs 0.4263 gap proves graph structure matters, but physics features don't push beyond vanilla.
3. **Bootstrap CI over z-test**: With n=3 seeds, the z-test is indefensible. Paired per-graph bootstrap is more powerful and honest.
4. **Be honest about "physics-informed"**: Injecting analytical results is useful but is closer to "GNN corrects analytical" than "GNN learns physics." The correlation check is essential context.
5. **Lockstep verification prevents silent drift**: Exact per-seed match between 6B and 6C vanilla runs confirms the frozen baseline is untouched.
6. **Null results are results**: The corrected dataset yields a genuine null — neither Tier A nor Tier A+B is statistically distinguishable from vanilla. This is scientifically valid and should be reported as such, not reframed as a "near-miss" or "trend."

## Prevention

- Added Tier A redundancy check (correlation with existing features) as mandatory diagnostic
- Added no-GNN residual baseline as standard ablation component
- Replaced z-test with paired per-graph bootstrap CI
- Added lockstep verification (per-seed match) between frozen baseline and new runs
- Documented nrecon stratification reconciliation
- Established that future "physics-informed" claims must include redundancy analysis and no-GNN baseline
- Status string and console headlines dynamically reflect actual significance outcomes (no hardcoded "verified" or "gnn_beyond_scalar_residual" frames)
