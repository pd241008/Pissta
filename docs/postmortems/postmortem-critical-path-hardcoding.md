# Postmortem: Critical-Path Tracker Hardcoding Bug

> **Date:** August 15, 2026  
> **Severity:** Medium — would have corrupted Stage 4+ results if undetected  
> **Status:** Fixed

## What Happened

The `monte_carlo.py` critical-path tracker hardcoded `G4` and `G5` as the merge predecessors:

```python
if AT[i, l_idx["G4"]] >= AT[i, l_idx["G5"]]:
    path1_critical[i] = True
else:
    path2_critical[i] = True
```

When we introduced the asymmetric sanity check DAG (with an extra gate `G7` on Path 2), the tracker still compared `G4` vs `G5`, completely ignoring `G7`. This produced a **50/50 split** instead of the expected **0/100 split**.

## Root Cause

- The tracker was written for the specific symmetric DAG, not generalized to any DAG topology.
- No abstraction existed to identify the sink node and its predecessors dynamically.
- The asymmetric sanity check (which should have caught this immediately) was added after the original code was written.

## Detection

The asymmetric sanity check (`experiments/asymmetric_sanity_check.py`) ran the modified DAG and reported:
```
Path 1 critical fraction: 0.4993
Path 2 critical fraction: 0.5007
FAIL: Path 2 does not dominate. Check AT/argmax logic.
```

This immediately flagged the bug.

## Fix

Generalized the tracker to:
1. Identify the sink node dynamically (`graph.sinks()`)
2. Find all predecessors of the sink
3. Use `np.argmax` on the predecessor arrival times to determine the critical path

```python
sinks = graph.sinks()
sink_preds = sorted([p for p, succs in graph.successors.items() if sink in succs])
winner_idx = np.argmax(pred_arrival, axis=1)
```

## Lessons Learned

1. **Never hardcode topology in analysis logic.** The timing graph should be abstract enough that adding/removing gates never requires changing the analysis engine.
2. **Sanity checks should test topology changes, not just parameter changes.** A symmetric DAG can hide hardcoding bugs.
3. **Dynamic predecessor detection is cheap.** The fix added negligible runtime but eliminated an entire class of bugs.

## Consequences if Undetected

- Stage 4 analytical SSTA would have been compared against corrupted MC labels.
- GNN surrogates (Stage 7) would have been trained on wrong critical-path data.
- Any paper results citing critical-path fractions would be invalid.
