# ADR-002: Clark MAX for Reconvergent Paths

> **Status:** Decided  
> **Date:** August 2026

## Context

At reconvergent nodes (e.g., G6), the critical-path delay is max(AT_G4, AT_G5), not a sum. We need an analytical approximation for this MAX operation that preserves correlation information from the shared upstream path.

## Options Considered

1. **Naive Gaussian (mean + 3σ)** — Ignores correlation and MAX nonlinearity. Underestimates tail.
2. **Clark's moment-matched MAX** — Approximates max(X,Y) as Gaussian matching first two moments. Preserves correlation through ρ.
3. **Full numerical convolution** — Exact but O(N²) or requires sampling. Defeats the purpose of analytical SSTA.
4. **Higher-order moment matching (e.g., Cornish-Fisher)** — Could capture skewness but adds complexity.

## Decision

We use **Clark's Gaussian moment-matched MAX approximation**.

## Reasoning

- Clark's formula is O(1) and requires only (μ1, σ1, μ2, σ2, ρ).
- It properly accounts for correlation between the two paths through shared gates and spatial variation.
- It is the standard textbook approximation for this problem in SSTA literature.
- The gap decomposition (Stage 4b) shows Clark captures ~67% of the MAX-induced tail expansion; the remaining ~33% comes from linearization error.

## Consequences

- Clark's P99.87 (10.445) is lower than the true MC (10.754) by 0.308.
- It is also lower than the naive Gaussian proxy (10.565) because it matches MAX moments more precisely.
- This is expected behavior, not a bug. The true MAX distribution has positive skewness that Gaussian moment-matching cannot capture.
