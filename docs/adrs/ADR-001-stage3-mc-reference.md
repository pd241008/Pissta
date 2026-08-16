# ADR-001: Stage 3 Monte Carlo as Ground Truth

> **Status:** Decided  
> **Date:** August 2026

## Context

We need a reference distribution for the critical-path delay of a branching timing graph under correlated process variation. This distribution will be the ground truth against which all later analytical SSTA, MAX approximations, and GNN surrogates are graded.

## Options Considered

1. **Analytical SSTA only** — Fast but unvalidated. No way to know if approximations are correct.
2. **Small Monte Carlo (N=10k)** — Fast but tail-sensitive. P99.87 has observable noise.
3. **Large Monte Carlo (N=100k)** — Slow but stable. Tail quantiles are well-converged.

## Decision

We use **N=100,000 Monte Carlo samples (seed 42)** as the locked ground truth.

## Reasoning

- N=10k is insufficient for tail convergence (delta P99.87 varies by ~0.06 across seeds).
- N=100k gives delta P99.87 stable to ±0.012 across 3 seeds.
- The 100k run takes ~200ms — acceptable for a one-time reference generation.
- Raw arrays are persisted to disk so Stage 4+ can load exact distributions without regenerating.

## Consequences

- All analytical SSTA comparisons must use the locked seed 42 reference.
- Regenerating from a different seed would silently break comparability.
- The 10k run remains available as a diagnostic-only artifact.
