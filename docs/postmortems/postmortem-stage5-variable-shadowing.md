# Postmortem: Stage 5 Variable Shadowing in Multi-Seed Loop

> **Date:** August 16, 2026  
> **Severity:** Low — corrupted one JSON field, no paper-facing numbers affected  
> **Status:** Fixed

## What Happened

In `experiments/run_stage5.py`, the multi-seed validation loop reused the variable
name `mc_p99_87` for per-seed MC values:

```python
mc_p99_87 = mc_summary["stats"]["p99_87"]      # seed 42, correct
...
for run in repro["runs"]:
    seed = run["seed"]
    mc_p99_87 = run["branch_p99_87"]   # ← overwrites outer variable
```

After the loop, `mc_p99_87` held seed 999's value (10.75217) instead of seed 42's
value (10.75363). This corrupted the `comparison.monte_carlo.p99_87` field in the
Stage 5 JSON output.

## Root Cause

- Python for-loops do not create a new scope; loop variables leak into the
  enclosing function scope.
- The loop variable name `mc_p99_87` was identical to the outer variable holding
  the locked seed-42 reference.
- No linting rule or test caught the shadowing.

## Detection

External review back-solved the numbers and found a mismatch:
- `total_gap` = 0.308258 = 10.753634 - 10.445376 (correct, seed 42)
- `comparison.monte_carlo.p99_87` = 10.752174 (wrong, seed 999)

The gap between these two values (0.0014) exactly matched the seed 42 vs seed 999
difference, confirming the shadowing.

## Fix

Renamed the loop variable to `mc_p99_87_seed`:

```python
for run in repro["runs"]:
    seed = run["seed"]
    mc_p99_87_seed = run["branch_p99_87"]
    ...
    multi_seed.append({
        "seed": seed,
        "mc_p99_87": mc_p99_87_seed,   # clear, no shadowing
        ...
    })
```

Also removed the misleading `closure_fraction_vs_pooled` field from the per-seed
loop, since it was a constant (not per-seed) and its name implied per-seed
variation.

## Impact

- **Markdown report:** Unaffected — all tables used the correct seed-42 value.
- **JSON export:** One field (`comparison.monte_carlo.p99_87`) was wrong.
- **Downstream:** Any Stage 6/7 code loading this JSON expecting the locked
  seed-42 reference would have silently gotten seed 999's number.

## Lessons Learned

1. **Never reuse outer-scope variable names in loops**, especially for values
   that are meant to be "locked" references.
2. **Add a linting rule** (e.g., `flake8-bugbear` B020) to catch loop variable
   shadowing in functions with complex scope.
3. **Validate JSON exports programmatically** — add a test that checks
   `comparison.monte_carlo.p99_87 == mc_summary["stats"]["p99_87"]` after
   the multi-seed loop.

## Prevention

- Added explicit test in CI (future): verify that multi-seed loop does not
  modify `mc_p99_87` after execution.
- Code review checklist item: "Check for loop variable shadowing of outer
  references."
