# Postmortem: Stage 6A Critical Issues — Physical Plausibility, Accounting, and Stats Mismatch

> **Date:** August 17, 2026  
> **Severity:** Critical — dataset unusable for GNN training without fixes  
> **Status:** Fixed and verified

## Issues Summary

Three critical issues were discovered during Stage 6A dataset validation, all rooted in insufficient physical bounds and incomplete accounting.

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
- `mean_delay`: max 28.6 (was 25,314)
- `std_delay`: max 1.10 (was 1,468,799)
- `graph_000024` noise: 0.06% (was 44.2%)
- All 20 validation graphs show < 0.1% mean noise, < 2% std noise

---

## Issue #2 — Missing Graph Accounting (652 Undocumented Skips)

### What Happened

The manifest reported `n_graphs=2000` but only 1,348 graphs were stored in `dataset.pkl`. The `splits.json` totals (943+202+203=1,348) were consistent with the dataset but not with the manifest. No explanation was provided for the missing 652 graphs.

### Root Cause

The `run_stage6a.py` pipeline silently skipped graphs where the sink had fewer than 2 predecessors (no MAX operation to learn), but:
- Did not track skip count
- Did not log skip reasons
- Reported `n_graphs=2000` in manifest without distinguishing generated vs stored

The 652 skipped graphs were all pure chains (0 reconvergence points), which is a separate structural finding that was hidden by the silent skip.

### Fix

In `data_generation/run_stage6a.py`:
- Added `skip_reasons` dict with explicit counters (`sink_predecessors_lt_2`, `mc_error`)
- Added try/except around MC generation to catch and log errors
- Changed manifest to report `target_n_graphs`, `n_generated`, `n_dataset` separately
- Added explicit console output: `"Skipped 652 graphs (sink with < 2 predecessors)"`

### Verification

Manifest now shows:
```json
{
  "target_n_graphs": 2000,
  "n_generated": 2000,
  "n_dataset": 1348,
  "skip_reasons": {
    "sink_predecessors_lt_2": 652,
    "mc_error": 0
  }
}
```

---

## Issue #3 — summary_stats.json Did Not Match Actual Dataset

### What Happened

`summary_stats.json` reported statistics computed from the pre-filter 2,000-graph set, not the actual 1,348-graph dataset:
- `reconvergence_points_per_graph.min = 0` (incorrect — actual dataset has min=1)
- `gates_per_graph.mean = 8.35` (incorrect — actual is 8.136)

This happened because `sizes` and `reconv_counts` were computed from the original `graphs` list before filtering, not from the surviving `dataset`.

### Root Cause

In `run_stage6a.py`, the summary stats block used `sizes` and `reconv_counts` variables that were computed from `graphs[:20]` (the validation subset of the pre-filter set), not from `dataset.values()`.

### Fix

In `data_generation/run_stage6a.py`:
- Compute `actual_sizes` and `actual_reconv` from `dataset.values()` after filtering
- Use these for summary stats instead of pre-filter variables
- Changed `n_graphs` in summary to `len(dataset)` instead of the original `n_graphs` parameter

### Verification

`summary_stats.json` now correctly reports:
```json
{
  "n_graphs": 1348,
  "gates_per_graph": {
    "min": 4, "max": 12, "mean": 8.136
  },
  "reconvergence_points_per_graph": {
    "min": 1, "max": 3, "mean": 1.332
  }
}
```

---

## Cross-Cutting Findings

### Pure-Chain Graphs Are Systematically Discarded

652/2000 (32.6%) of generated graphs are pure chains with 0 reconvergence points. This is a property of the generator's split-reconverge construction:
- Subdivision adds 1 gate but doesn't create reconvergence
- Split-reconverge adds 2+ gates and creates reconvergence
- With n_gates ~ Uniform(4,12), many small graphs are pure chains

**Implication**: The generator is biased toward small pure chains. For Stage 6B, this means the training set is skewed toward simpler topologies. Consider:
- Increasing `n_gates_range` minimum to 6
- Adding a "must have ≥1 reconvergence" constraint to the generator
- Or accepting the bias and ensuring the test set has sufficient complex graphs

### Label Noise Validation Was Itself Contaminated

The initial label noise validation (17% mean noise, 12,926% std noise) was computed on graphs that included the outlier `graph_000024`. After fixing the singularity:
- Mean noise: 0.06% ± 0.04%
- Std noise: 0.62% ± 0.47%

This confirms the fix resolved the underlying instability, not just filtered symptoms.

---

## Lessons Learned

1. **Physical bounds are non-negotiable**: Any sampler that feeds a nonlinear physical model must enforce hard bounds at the source, not hope the downstream model handles singularities gracefully.
2. **Account for every data point**: Silent skips in data pipelines hide structural biases. Always log what was discarded and why.
3. **Stats must come from the final dataset**: Computing summary statistics from intermediate data structures leads to mismatches that erode trust in the entire dataset.
4. **Validate validation data**: The label noise check itself was contaminated by the same bug it was supposed to detect. Always verify the validator.

## Prevention

- Added physical bounds enforcement in `variation/sampler.py` (Vth clipping, Pelgrom cap)
- Added explicit skip accounting in `data_generation/run_stage6a.py`
- Added summary stats computation from `dataset.values()` only
- Future: Add CI test that checks `summary_stats.json` fields match actual dataset on load
