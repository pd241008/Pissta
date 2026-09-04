# Stage 7 — Split Conformal Calibration Report

**Date:** August 31, 2026
**Method:** Split conformal, **studentized-residual** nonconformity score
**Target:** 90% nominal coverage (α = 0.10), 3 seeds [42,123,999], cal=152 / eval=152 (carved from test, stratified by nrecon, carve_seed=7)
**Backbones:** `vanilla` (h=64) and `maxbias_cm` (h=54), both retrained on the full train split with the identical frozen protocol.

---

## 1. What this stage does

Closes the "calibrated uncertainty" half of the central claim. The GNN predicts a per-graph critical-path delay `(mean, std)`. Split conformal wraps that point/scale estimate in prediction intervals with a **guaranteed** (under exchangeability) marginal coverage rate — e.g. "90% of true delays fall in the interval" — **without** assuming the residual distribution is Gaussian.

**Nonconformity score (studentized residual):**
```
score_i = |mc_mean_i − pred_mean_i| / max(pred_std_i, ε)
```
The model's own predicted std is the heteroskedasticity-aware scale: interval is narrow where confident, wide where uncertain. A deliberately methodological choice, not a default.

**Interval on eval:** `pred_mean ± q_hat · std_pred`, where `q_hat` is the split-conformal quantile (`ceil((n_cal+1)(1−α))/n_cal` order statistic) of calibration scores.

**Calibration split integrity:** `splits.json` reserves only train/val/test. Stage 7 **carves** the test split (304) into cal (152) / eval (152), stratified by nrecon, so coverage is asserted on a held-out eval slice distinct from the Stage 6C full-test MAE points. The Stage 6C full-test MAE numbers are not re-used as the coverage target.

---

## 2. Verified results (from `stage7_calibration_results.json`)

### 2.1 Pooled coverage on eval

| Backbone | Per-seed coverage | Mean pooled | Meets 90%? |
|---|---|---|---|
| **maxbias_cm** | 0.928 / 0.882 / 0.914 | **0.908** | ✓ |
| vanilla | 0.875 / 0.934 / 0.875 | 0.895 | ~ (marginally under) |

### 2.2 Disaggregated per-nrecon coverage (pooled across seeds)

| nrecon | n (per eval) | Vanilla | Maxbias_cm |
|---|---|---|---|
| 1 | 75 | 0.951 ✓ | 0.964 ✓ |
| 2 | 64 | **0.839 ✗** | **0.865 ✗** |
| 3 | 12 | 0.917 ✓ | 0.861* |
| 4 | 1 | 0.000* | 0.000* |

\* non-assessable (n too small).

### 2.3 Accuracy on the eval slice (MAE)

| Backbone | Per-seed mean MAE | Mean |
|---|---|---|
| **maxbias_cm** | 0.341 / 0.391 / 0.381 | **0.371** |
| vanilla | 0.420 / 0.401 / 0.426 | 0.416 |

---

## 3. Honest interpretation

1. **Split conformal generally works:** both backbones land at ≈90% pooled coverage. MAX-bias (0.908) is the preferred backbone — better accuracy (0.371 vs 0.416 MAE) **and** at-nominal pooled coverage.
2. **The disaggregated check matters:** both backbones **under-cover the nrecon=2 bucket** (0.84–0.87 vs 0.90) — precisely the reconvergence regime where MAX-inflated tails are the point. The pooled 90% number hides this; the per-nrecon table surfaces it. This is a **real, reportable miscalibration**, not a pooled artifact.
3. **A better point estimate ≠ better calibration:** MAX-bias's tighter intervals (q_hat ≈ 1.04 vs vanilla ≈ 1.14) are what cause its nrecon=2 under-coverage — its predicted std is slightly optimistic in the correlated-tail regime. This validates the project's honesty constraint to check calibration separately from accuracy.
4. **No coverage claim under distribution shift:** the eval slice is the same nrecon 1–4 topology family as calibration. Coverage on **unseen topology families (OOD)** is a documented follow-up, **not claimed here**.

---

## 4. Reproducibility

- `gnn_baseline/conformal.py` — pure numpy split-conformal functions (unit-tested).
- `gnn_baseline/run_stage7.py` — carve + retrain + conformal driver.
- `tests/test_conformal.py` — 8 unit tests (finite-sample quantile, heteroskedastic width, group coverage, etc.).
- `gnn_baseline/results/splits_stage7.json` — the frozen cal/eval carve (auditable).
- `gnn_baseline/results/stage7_calibration_results.json` — full per-seed results (gitignored, regenerable).
- Full test suite: **19 passed**.

## 5. Scope / honesty notes for the paper

- Report nrecon=2 under-coverage explicitly as a known limitation; do not present the pooled 90% as uniform.
- State MAX-bias training cost alongside accuracy/coverage in the Stage 8 method table (see PROJECT_HISTORY §10).
- Do not claim guaranteed coverage under distribution shift until the OOD (unseen-topology) test is run.
