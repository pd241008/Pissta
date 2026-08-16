# ADR-003: Analytical SSTA with Linearized Delay Model

> **Status:** Decided  
> **Date:** August 2026

## Context

We need an analytical SSTA pipeline that can compute timing moments without Monte Carlo sampling. The delay model is the alpha-power law with geometry sensitivity, which is nonlinear in Vth, L, and W.

## Options Considered

1. **Full nonlinear analytical** — Intractable. Alpha-power with geometry factor has no closed-form moments.
2. **First-order Taylor linearization** — Linearize delay around nominal, propagate Gaussian moments analytically.
3. **Second-order Taylor (delta method)** — Better variance estimate but more complex covariance bookkeeping.
4. **Moment matching with sampling** — Use MC to fit moments, then interpolate. Still requires sampling.

## Decision

We use **first-order Taylor linearization** of the alpha-power delay model.

## Reasoning

- First-order linearization is the standard approach in SSTA (e.g., ASPDAC 2004+ literature).
- It yields closed-form expressions for ∂d/∂Vth, ∂d/∂L, ∂d/∂W.
- Covariance propagation is straightforward: Cov(d_g, d_h) = Σ_j (∂d_g/∂X_j)(∂d_h/∂X_j) Cov(X_j^g, X_j^h).
- The gap decomposition shows linearization accounts for ~33% of the total error (0.100 of 0.308).

## Consequences

- Per-gate delay variances are underestimated by ~3–5% compared to empirical MC.
- Tail percentiles (P99.87) are further underestimated because the linearized delay distribution is closer to Gaussian than the true nonlinear distribution.
- This is a known, documented limitation — not a bug.
