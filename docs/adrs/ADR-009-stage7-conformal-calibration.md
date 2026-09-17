# ADR-009: Stage 7 Split Conformal Calibration of the DAG-GNN Surrogate

> **Status:** Decided  
> **Date:** August 31, 2026  
> **Last updated:** August 31, 2026

## Context

Everything through Stage 6C delivered **point/scale estimates** (per-graph critical-path delay mean and std MAE) but not a **calibrated uncertainty interval**. The paper's thesis is "physics + statistical timing + ML surrogate + **calibrated uncertainty**" — calibration is the half that has not been touched. The MAX-bias GNN (capacity-matched, h=54) is the verified-strongest backbone; Stage 7 wraps it in a distribution-free conformal procedure that yields prediction intervals with a **guaranteed** (under exchangeability) marginal coverage rate, e.g. "90% of true delays fall in this interval," WITHOUT assuming the residual distribution is Gaussian.

The project's honesty constraints bite hardest here:
- Do **not** claim coverage under arbitrary distribution shift (e.g. unseen topology families) unless actually tested.
- Report **tail error** and **disaggregated** coverage (per-topology bucket), not just pooled — a model well-calibrated on average but miscalibrated on complex topologies is a real failure the pooled number hides.

## Options Considered

1. **Split conformal + studentized residual** — Calibrate on a held-out calibration set, nonconformity score `|y_true − mean_pred| / std_pred`, interval `mean_pred ± q_hat·std_pred`. Simple, one calibration pass, fully auditable, and the model's own predicted std acts as a heteroskedasticity-aware scale (narrow intervals where confident, wide where uncertain). Matches the existing mean/std output head with no architecture change.
2. **Conformalized quantile regression (CQR)** — Would require changing the model head to output quantiles, a substantially larger lift, and only adds value if per-quantile calibration is specifically needed.

## Decision

We use **split conformal with the studentized-residual nonconformity score** (option 1), consistent with the project's preference for the simpler, well-verified approach.

**Calibration split:** the existing `splits.json` reserves train/val/test only — there is **no dedicated calibration split**. We carve the **test** split (304 graphs) into `cal` (152) and `eval` (152), **stratified by nrecon** (75/64/12/1 each), deterministic on `carve_seed=7`, so the Stage 7 coverage is asserted on a held-out **eval** slice rather than the same graphs whose full-test MAE Stage 6C already reported. The Stage 6C full-test MAE numbers are not re-used as the Stage 7 coverage target.

- **Score:** `score_i = |mc_mean_i − pred_mean_i| / max(pred_std_i, ε)`
- **Quantile:** `q_hat` = the `ceil((n_cal+1)(1−α))/n_cal` order statistic of calibration scores (standard split conformal, finite-sample corrected).
- **Interval on eval:** `pred_mean ± q_hat · pred_std`
- **Target:** α = 0.10 → **90% nominal coverage**, checked pooled and per-nrecon.
- **Backbones:** retrain both `vanilla` (h=64) and `maxbias_cm` (h=54) fresh on the full train split with the identical frozen protocol (seeds [42,123,999]) so the calibration comparison is fair and reproducible. (The MAX-bias CM checkpoints were not on disk — they had to be re-trained; vanilla was also retrained so both are identically provisioned.)

## Verified Results (90% nominal, 3 seeds, cal=152 / eval=152)

**Pooled coverage on eval:**
- **Maxbias_cm: 0.908** (per-seed 0.928/0.882/0.914) — meets 90% nominal.
- **Vanilla: 0.895** (per-seed 0.875/0.934/0.875) — marginally under 90%.

**Disaggregated per-nrecon coverage (pooled across seeds):**

| nrecon | n (per eval) | Vanilla | Maxbias_cm | Finding |
|---|---|---|---|---|
| 1 | 75 | 0.951 | 0.964 | both meet target |
| 2 | 64 | **0.839** | **0.865** | **both under-cover** |
| 3 | 12 | 0.917 | 0.861* | small n |
| 4 | 1 | 0.000* | 0.000* | n=1, not assessable |

*small / non-assessable cell.

**Eval MAE:** Maxbias_cm 0.371 (0.341/0.391/0.381) vs Vanilla 0.416 (0.420/0.401/0.426) — the architectural win reconfirms on the fresh held-out eval slice.

**Interpretation (honest):**
- Both backbones hit ≈90% **pooled** coverage — split conformal works marginally.
- **Both systematically under-cover the nrecon=2 bucket** (0.84–0.87 vs 0.90) — the reconvergence regime where MAX-inflated tails are the point. This is the disaggregated failure mode the pooled number hides.
- **MAX-bias's tighter intervals (q_hat ≈ 1.04 vs vanilla ≈ 1.14) are what produce its nrecon=2 under-coverage**: its predicted std is slightly optimistic in the correlated-tail regime. A better point estimate (lower MAE) does **not** automatically imply better calibration — the very wedge the user flagged.
- Coverage under **distribution shift to unseen topology families is NOT asserted**; the eval slice is the same nrecon 1–4 family as calibration. An OOD test (unseen topology family) is scoped as a documented follow-up, not claimed.

## Consequences

- **Positive:** the surrogate now ballparks genuinely (≈90%) calibrated uncertainty intervals, with MAX-bias the preferred backbone (better accuracy AND at-nominal pooled coverage).
- **Caveat (must be in paper):** nrecon=2 under-coverage is a real, non-pooled miscalibration for **both** backbones. Improving calibration in the MAX-heavy regime (e.g. per-bucket conformal quantiles, or a wider std head) is future work, not silently claimed away.
- **Runtime/cost:** MAX-bias CM training cost across 3 seeds is on the order of the vanilla training cost; inference is unchanged from Stage 6C. Stage 8's comparison table must state this cost alongside the accuracy/coverage (see `results/stage8_report.md`) rather than presenting the intervals as free.
- Files: `gnn_baseline/conformal.py`, `gnn_baseline/run_stage7.py`, `tests/test_conformal.py`, `gnn_baseline/results/splits_stage7.json`, `gnn_baseline/results/stage7_calibration_results.json` (results gitignored — the committed record is this ADR + report).
