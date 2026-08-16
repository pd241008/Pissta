# ADR-005: Stage 5 Tail-Aware Reporting — Pooled MC Reference and Absolute Gaps

> **Status:** Decided  
> **Date:** August 2026

## Context

Stage 5 introduced a skew-normal 3-moment MAX approximation to capture tail skew.
Initial reporting compared the tail-aware estimate against individual Monte Carlo
seeds and reported closure percentages that varied across seeds (54.28% → 62.43%).
This variation was entirely due to MC reference noise, not tail-aware method
instability — but the reporting made it look like the method's performance was
seed-dependent.

## Problem

Closure% = (MC_P99.87 - Tail_P99.87) / shape_gap is a ratio whose denominator
is a stochastic MC estimate. When the denominator varies by ±0.012 (Stage 3's
characterized seed-to-seed noise), the ratio swings by ±4–5 percentage points.
This creates a false impression of method instability and makes it impossible to
compare against other work.

## Options Considered

1. **Report closure% vs seed 42 only** — Simple but hides the fact that seed 42
   is just one noisy draw.
2. **Report mean/std of absolute gaps across seeds** — More honest but still
   leaves the reader to compute closure% themselves.
3. **Pooled MC reference + absolute gaps as primary, closure% as secondary** —
   Combines the best of both: deterministic headline number with transparent
   per-seed detail.

## Decision

We use **pooled MC reference + absolute gaps as primary metrics**, with closure%
reported as secondary context only.

## Reasoning

- Pooled MC P99.87 = mean(10.7536, 10.7367, 10.7522) = 10.7475 is a more
  stable reference than any single seed.
- Absolute gap (mean 0.0891 ± 0.0090) is dimensionally honest and doesn't get
  inflated/deflated by which seed's MC estimate happens to be the denominator.
- Closure% vs pooled MC (56.9%) is the single headline figure; the 54–62% range
  across individual seeds is explicitly documented as MC reference noise.
- This matches the tightened reporting discipline introduced in Stage 5.

## Consequences

- All Stage 5+ comparisons must use pooled or absolute-gap metrics, not
  closure% vs individual seeds.
- Stage 4/4b numbers remain valid (they don't report closure%).
- Future stages (6, 7, 8) inherit this reporting standard.
