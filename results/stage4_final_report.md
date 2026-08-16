# Stage 4 Final Report — Analytical SSTA for Branching Timing Graph

**Date:** 2026-08-15  
**Stage:** Stage 4 — Analytical SSTA (Clark MAX Approximation)  
**Status:** Complete — covariance bookkeeping verified, analytical error is consistent across seeds

---

## 1. Objective

Replace Monte Carlo sampling with closed-form analytical moments for the same
branching DAG and variation model defined in `stage3_config.json`. The analytical
pipeline must produce mean/std/percentiles that can be directly compared against
the locked Stage 3 reference distribution.

---

## 2. Pipeline Summary

| Step | Module | What it computes |
|------|--------|------------------|
| 1 | `analytical_variation.py` | Closed-form mean/var/cov of L, W, Vth at each gate |
| 2 | `analytical_delay.py` | Linearized alpha-power delay moments (mean/var/cov per gate) |
| 3 | `statistical_sum.py` | Sequential sum propagation along paths (AT_G4, AT_G5) |
| 4 | `clark_max.py` | Clark's Gaussian moment-matched MAX approximation |
| 5 | `stage4_analytical_ssta.py` | Final sum with G6, percentiles, comparison vs Stage 3 |

---

## 3. Analytical Moments

### 3.1 Process Moments (L, W, Vth)

For each gate i:
- **Mean:** μ_Xi = X_nom (inter-die and spatial/random terms are zero-mean)
- **Variance:** Var(X_i) = σ²_inter + σ²_spatial(i,i) + σ²_random
- **Covariance:** Cov(X_i, X_j) = σ²_inter + σ²_spatial(i,j)

The spatial covariance uses the distance-based exponential model:
Σ_ij = σ² exp(-d_ij / λ)

with gate coordinates from `stage3_config.json`.

### 3.2 Linearized Delay Model

Alpha-power delay: d = C_load·Vdd / [k·(Vdd−Vth)^α] · (L/L_nom) / √(W/W_nom)

First-order Taylor expansion around nominal:
- ∂d/∂Vth = d_nom · α / (Vdd − Vth_nom)
- ∂d/∂L = d_nom / L_nom
- ∂d/∂W = −d_nom / (2·W_nom)

Delay variance for gate g:
σ²_dg = Σ_j (∂d_g/∂X_j)² Var(X_j)

Delay covariance between gates g and h:
Cov(d_g, d_h) = Σ_j (∂d_g/∂X_j)(∂d_h/∂X_j) Cov(X_j,g, X_j,h)

(Cross-parameter covariances are zero by construction.)

### 3.3 Path Propagation (Sum)

For two Gaussian variables X ~ N(μ_X, σ²_X) and Y ~ N(μ_Y, σ²_Y) with Cov(X,Y) = σ_XY:

X + Y ~ N(μ_X + μ_Y, σ²_X + σ²_Y + 2σ_XY)

Applied sequentially along:
- **Path 1:** AT_G4 = d_G1 + d_G2 + d_G4
- **Path 2:** AT_G5 = d_G1 + d_G3 + d_G5

### 3.4 Clark MAX Approximation

Given X ~ N(μ1, σ1²) and Y ~ N(μ2, σ2²) with correlation ρ:

a = √(σ1² + σ2² − 2ρσ1σ2)
θ = (μ1 − μ2) / a

μ_max = μ1 Φ(θ) + μ2 Φ(−θ) + a φ(θ)
σ²_max = (μ1²+σ1²)Φ(θ) + (μ2²+σ2²)Φ(−θ) + (μ1+μ2)a φ(θ) − μ_max²

Φ = standard normal CDF, φ = standard normal PDF.

### 3.5 Final Sum with G6

μ_final = μ_max + μ_d6
σ²_final = σ²_max + σ²_d6 + 2 · Cov(max, d6)

where Cov(max, d6) ≈ Φ(θ) · Cov(AT_G4, d6) + Φ(−θ) · Cov(AT_G5, d6)

---

## 4. Analytical Results

### 4.1 Path Arrival Times at Reconvergence

| Node | Mean | Std | Notes |
|------|------|-----|-------|
| AT_G4 | 6.4109 | 0.2789 | Path 1 (G1→G2→G4) |
| AT_G5 | 6.4109 | 0.2789 | Path 2 (G1→G3→G5) |
| Cov(AT_G4, AT_G5) | 0.0281 | — | Positive due to shared G1 + spatial correlation |
| ρ(AT_G4, AT_G5) | 0.3618 | — | Moderate positive correlation |

### 4.2 Clark MAX Approximation

| Quantity | Value |
|----------|-------|
| μ_max | 6.5366 |
| σ_max | 0.2490 |

### 4.3 Final Critical-Path Delay (After Adding G6)

| Metric | Analytical | Monte Carlo (Seed 42) | Absolute Error | Relative Error |
|--------|------------|----------------------|----------------|----------------|
| Mean | 9.2564 | 9.3065 | −0.0501 | −0.54% |
| Std | 0.3963 | 0.4194 | −0.0231 | −5.50% |
| P95 | 9.9083 | 10.0243 | −0.1159 | −1.16% |
| P99 | 10.1782 | 10.3675 | −0.1892 | −1.83% |
| P99.87 | 10.4454 | 10.7536 | **−0.3083** | **−2.87%** |

---

## 5. Runtime Comparison

| Method | Runtime | Notes |
|--------|---------|-------|
| Analytical SSTA | **0.35 ms** | Deterministic, single evaluation |
| Monte Carlo (N=100k) | **200 ms** | Includes sampling + timing propagation |

**Speedup: ~566×**

The analytical method is near-instant and requires no sampling. This is the first
real speedup number for the paper's results table.

---

## 6. Multi-Seed Validation

Since analytical SSTA is deterministic given the config, there is only one analytical
output. We validate stability by comparing it against all three Stage 3 Monte Carlo
seeds:

| Seed | MC P99.87 | Analytical P99.87 | Absolute Error |
|------|-----------|-------------------|----------------|
| 42 | 10.7536 | 10.4454 | −0.3083 |
| 123 | 10.7367 | 10.4454 | −0.2913 |
| 999 | 10.7522 | 10.4454 | −0.3068 |

**Conclusion:** The analytical error is consistent across all three MC seeds
(range: −0.291 to −0.308, std ≈ 0.009). This confirms the error is a property
of the analytical approximations, not an artifact of a particular seed.

---

## 7. Error Direction Sanity Check

The user's instruction stated: *"If Clark's P99.87 comes out worse than the naive
gaussian+3σ approximation, stop and check your covariance bookkeeping."*

- **Naive Gaussian (Stage 3 MC mean + 3σ):** 9.3065 + 3×0.4194 = **10.5648**
- **Clark P99.87:** **10.4454**

Clark's result is indeed lower by ≈0.12. **We checked covariance bookkeeping and
verified it is correct:** analytical per-gate delay covariances match empirical
Monte Carlo covariances within 3–5%. The "worse" result is expected, not a bug.

### Why Clark is Lower Than Naive Gaussian

The naive Gaussian approximation uses the **full Monte Carlo** mean and std, which
include all higher-order nonlinear effects from the alpha-power delay model and the
MAX operation. The analytical pipeline makes two approximations:

1. **Linearized delay model:** Neglects higher-order Taylor terms in the delay
   nonlinearity. This systematically underestimates per-gate delay variance.
2. **Clark MAX + Gaussian final sum:** Clark's formula matches the first two moments
   of the MAX exactly, but the true MAX distribution is skewed with a heavier tail
   than a Gaussian with the same moments. The subsequent sum with G6 also assumes
   Gaussianity.

Together, these approximations produce a distribution that is closer to Gaussian
than the true Monte Carlo distribution, so its 3σ quantile (10.445) falls short of
both the true MC tail (10.754) and the naive Gaussian proxy based on MC stats (10.565).

This is a **known limitation of first-order analytical SSTA**, not a covariance bug.

---

## 8. Covariance Bookkeeping Verification

We compared analytical delay covariances against empirical Monte Carlo covariances
from `stage3_raw.npz` (N=100k, seed 42):

| Gate | Analytical Var | Empirical Var | Ratio |
|------|---------------|---------------|-------|
| G1 | 0.00988 | 0.00997 | 0.991 |
| G2 | 0.02053 | 0.02096 | 0.980 |
| G3 | 0.02053 | 0.02091 | 0.982 |
| G4 | 0.03464 | 0.03592 | 0.964 |
| G5 | 0.03464 | 0.03540 | 0.979 |
| G6 | 0.07493 | 0.07922 | 0.946 |

The analytical covariances are consistently within ~3–5% of empirical values.
Cross-gate covariances (e.g., Cov(G4,G6), Cov(G5,G6)) also match within the same
tolerance. **No bookkeeping bugs detected.**

---

## 9. Key Takeaways

1. **Analytical SSTA is 566× faster** than N=100k Monte Carlo (0.35 ms vs 200 ms).
2. **Covariance bookkeeping is verified correct** — analytical and empirical
   delay covariances agree within ~3–5%.
3. **Clark's P99.87 (10.445) is lower than naive Gaussian (10.565)**, but this is
   expected due to first-order linearization and Gaussian approximations, not a bug.
4. **Error is consistent across all three MC seeds** (−0.291 to −0.308), confirming
   it is an inherent limitation of the analytical method, not seed-dependent noise.
5. **The analytical method is useful** for rapid exploration and floorplanning
   sensitivity analysis, but for sign-off-quality tail estimates, Monte Carlo
   (Stage 3) remains the reference.

---

## 10. Files

| File | Description |
|------|-------------|
| `analytical_variation.py` | Closed-form process parameter moments |
| `analytical_delay.py` | Linearized delay moments |
| `statistical_sum.py` | Gaussian sum propagation |
| `clark_max.py` | Clark's MAX approximation |
| `stage4_analytical_ssta.py` | Orchestration + comparison |
| `results/stage4_analytical_ssta.json` | Full report with all numbers |

---

## 11. Next Steps

- Stage 5: Implement Clark-style MAX directly in the timing graph (not just at
  the final reconvergence) to handle multi-level reconvergent DAGs.
- Stage 6: Evaluate whether higher-order Taylor terms or moment-matching can
  improve tail accuracy without returning to full Monte Carlo.
- Stage 7: GNN surrogate trained on Stage 3 reference, validated against both
  analytical and MC distributions.
