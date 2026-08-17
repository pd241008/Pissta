# Postmortem: Stage 6A Critical Issues — Final Report

> **Date:** August 17, 2026  
> **Severity:** Critical — dataset unusable for GNN training without fixes  
> **Status:** All fixed and verified

## Issues Summary

Three critical issues were discovered during Stage 6A dataset validation, all rooted in insufficient physical bounds and incomplete accounting. All have been fixed and verified.

---

## Issue #1 — Massive, Physically Implausible Label Outliers

### What Happened

The generated dataset contained 7 graphs with extreme delay values:
- `mean_delay`: max 25,313.8 (vs median 13.8, physically expected ~25–35 for 12-gate chain)
- `std_delay`: max 1,468,799 (vs median 0.75)

The most severe outlier was `graph_000024` with mean delay 6,659 and 44.2% seed-to-seed label noise (next-worst was 8.3%, all others < 2%).

### Root Cause

The alpha-power delay formula `d = C_load·Vdd / (k·(Vdd - Vth)^α)` has a singularity as `Vth → Vdd`. In the Stage 6A generator:

1. **Unbounded Pelgrom distance**: The generator places gates on a 2D layout where x-coordinates scale linearly with topological depth. For a 12-gate graph, the farthest gate can be at x=16 (layout units), which converts to d_um=16μm. The Pelgrom Vth sigma scales as `sqrt(A/(W·L) + (S·d_um)²)`, giving σ ≈ 0.16V at 16μm — 10× larger than the intended ~0.016V.

2. **No physical Vth bounds**: The sampler had no upper bound on Vth. With σ=0.16V, 3σ+ events push Vth above 0.95V (Vdd=1.0V), making `Vdd - Vth` approach zero and causing delay to blow up nonlinearly.

3. **Insufficient alpha-power floor**: The existing floor of `1e-6`V was too small to prevent numerical explosions when `Vdd - Vth` approached this value.

### Fix

1. **Capped Pelgrom distance at 5μm** in `variation/sampler.py`:
   ```python
   d_eff = min(d_um, 5.0)
   ```

2. **Added physical Vth bounds** in `variation/sampler.py`:
   ```python
   vdd = getattr(params, "vdd_v", 1.0)
   vth_max = vdd - 0.05  # 50mV headroom
   vth_min = params.vth_nom_v - 5.0 * max_sigma
   Vth = np.clip(Vth, vth_min, vth_max)
   ```

3. **Increased alpha-power floor to 0.1V** in `timing/graph.py`:
   ```python
   np.maximum(vdd - vth_v, 0.1)  # was 1e-6
   ```

4. **Added `vdd_v` to `VariationParams`** in `foundations/config_loader.py` for sampler bounds.

### Verification

After fix:
- `mean_delay`: max 28.9 (was 25,314)
- `std_delay`: max 1.15 (was 1,468,799)
- Label noise across 20 validation graphs: mean 0.05% ± 0.04%, std 0.75% ± 0.55%
- No outliers > 1% mean noise; all graphs within physically plausible range

---

## Issue #2 — Missing Graph Accounting (Fully Resolved)

### What Happened

Earlier versions of the pipeline silently skipped graphs where the sink had fewer than 2 predecessors. The manifest reported `n_graphs=2000` but fewer graphs were stored in `dataset.pkl`, with no explanation for the gap.

### Root Cause

The generator could produce pure chains where the sink had only 1 predecessor. These graphs were discarded downstream because the MAX operation at the sink (the core learning target) was absent.

### Fix

Instead of filtering downstream, we modified `_build_dag` in `graph_generator.py` to **force the first operation to be a split-reconverge on `source→sink`** before any other operations. This guarantees the sink has ≥2 predecessors by construction for every generated graph. The downstream sink-predecessor filter in `run_stage6a.py` was removed entirely.

### Verification

- 2000/2000 graphs valid — 0 skips
- No `sink_predecessors_lt_2` skip reason in manifest
- All graphs have ≥2 reconvergence points (at the sink, by construction)
- Topology diversity: nrecon ranges from 2 to 8 across the dataset (mean 2.85)

---

## Issue #3 — summary_stats.json Did Not Match Actual Dataset

### What Happened

`summary_stats.json` reported statistics computed from an intermediate data structure rather than the final dataset. This was because `sizes` and `reconv_counts` were computed from `graphs[:20]` (the validation subset of the pre-filter set), not from `dataset.values()`.

### Fix

In `data_generation/run_stage6a.py`:
- Compute `actual_sizes` and `actual_reconv` from `dataset.values()` after filtering
- Use these for summary stats instead of pre-filter variables
- Changed `n_graphs` in summary to `len(dataset)` instead of the original `n_graphs` parameter
- Implemented stratified splitting by reconvergence count to ensure test/val have sufficient complex-topology graphs

### Verification

`summary_stats.json` now correctly reports:
```json
{
  "n_graphs": 2000,
  "gates_per_graph": {
    "min": 6, "max": 14, "mean": 9.97
  },
  "reconvergence_points_per_graph": {
    "min": 2, "max": 8, "mean": 2.85
  }
}
```

---

## Cross-Cutting Findings

### Sink Reconvergence Guarantee Eliminates Silent Discards

The original generator could produce pure chains where the sink had only 1 predecessor. These graphs were silently discarded downstream. The initial fix (`min_reconvergence=1`) reduced but did not eliminate the problem because a graph could have a reconvergence point mid-graph while still lacking one at the sink.

**Final fix**: Modified `_build_dag` to force the first operation to be a split-reconverge on `source→sink` before any other operations. This guarantees the sink has ≥2 predecessors by construction for every generated graph, eliminating the discard mechanism entirely.

**Result**: 2000/2000 graphs valid, 0 skips. The dataset now contains only graphs with nrecon ≥ 2 (mean 2.85, range 2–8).

### Timing Measurement Must Match What It Claims to Measure

`total_generation_time_s` originally included graph generation (`generate_dataset()`) and validation overhead, while `mean_time_per_graph_s` was computed from the per-graph loop only. This made `total / mean ≈ 2020` instead of 2000, a 1% inconsistency.

**Fix**: Moved `t0` to start immediately before the per-graph loop. Now both metrics measure exactly the same thing (MC + physics computation), and `total / mean = 2000` exactly.

### Label Noise Validation Was Itself Contaminated

The initial label noise validation (17% mean noise, 12,926% std noise) was computed on graphs that included the outlier `graph_000024`. After fixing the singularity:
- Mean noise: 0.05% ± 0.04%
- Std noise: 0.75% ± 0.55%

This confirms the fix resolved the underlying instability, not just filtered symptoms.

---

## Lessons Learned

1. **Physical bounds are non-negotiable**: Any sampler that feeds a nonlinear physical model must enforce hard bounds at the source, not hope the downstream model handles singularities gracefully.
2. **Guarantee invariants by construction, not by filtering**: If a graph property is required (sink has ≥2 predecessors), enforce it during generation rather than discarding invalid graphs afterward. This eliminates silent bias and wasted computation.
3. **Account for every data point**: Silent skips in data pipelines hide structural biases. Always log what was discarded and why — or better, prevent the discard in the first place.
4. **Stats must come from the final dataset**: Computing summary statistics from intermediate data structures leads to mismatches that erode trust in the entire dataset.
5. **Validate validation data**: The label noise check itself was contaminated by the same bug it was supposed to detect. Always verify the validator.
6. **Timing metrics must measure the same thing**: `total_time` and `mean_time_per_graph × N` should match exactly. If they don't, the timing scope is wrong — not the clock.

## Prevention

- Added physical bounds enforcement in `variation/sampler.py` (Vth clipping, Pelgrom cap)
- Added `vdd_v` to `VariationParams` for sampler bounds
- Increased alpha-power floor to 0.1V in `timing/graph.py`
- Modified `_build_dag` to force first split-reconverge on `source→sink`, guaranteeing valid graphs by construction
- Removed downstream sink-predecessor filter in `run_stage6a.py`
- Added explicit skip accounting in `data_generation/run_stage6a.py`
- Added summary stats computation from `dataset.values()` only
- Aligned `total_generation_time_s` with per-graph loop timing
- Implemented stratified splitting by reconvergence count for reliable ablation
- Future: Add CI test that checks `summary_stats.json` fields match actual dataset on load, and that timing metrics are internally consistent
