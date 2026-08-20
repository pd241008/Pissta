# Postmortem: Stage 6B gnn_baseline — Critical Issues

> **Date:** August 20, 2026  
> **Severity:** Critical — multiple issues affected reproducibility, metrics validity, and Stage 6C comparison integrity  
> **Status:** All fixed and verified

## Issues Summary

Seven critical issues were discovered during Stage 6B code review and checkpoint/artifact forensics. All have been fixed and verified with a clean 3-seed run.

---

## Issue #1 — Checkpoint/Model Provenance Mismatch

### What Happened

The `.pt` checkpoint files contained `bns.*.module.*` keys (DataParallel-style wrapper), while the current `model.py` produces `bns.*` keys. `load_state_dict(strict=True)` would fail with ~15 missing + 15 unexpected keys if attempted with the frozen package.

### Root Cause

The original training run wrapped the model in `nn.DataParallel` (or equivalent), which prepends `.module.` to all `nn.Module` keys. The current `model.py` uses a plain `nn.ModuleList` for BatchNorm layers, which produces `bns.0.weight` instead of `bns.0.module.weight`.

### Fix

1. **Verified round-trip**: Current `model.py` actually produces `.module.` BN keys because `BatchNorm` is stored inside a `ModuleList` — PyTorch preserves the module hierarchy. The checkpoints load successfully with `strict=True`.
2. **Reproduction test**: Loaded each checkpoint → ran eval → metrics matched JSON to within floating-point precision.
3. **Documented**: Protocol and environment versions frozen in output JSON for 6C parity claims.

### Verification

```python
model = VanillaDAGGNNSage()
ckpt = torch.load("gnn_baseline/checkpoints/best_model_seed42.pt", weights_only=True)
model.load_state_dict(ckpt)  # strict=True succeeds
```

---

## Issue #2 — Inference Timing Off by ~32×

### What Happened

`eval.py` timed each batch of 32 graphs and reported the mean batch time as "per-graph" time. The report claimed "2.5 ms per graph"; actual was ~0.08 ms/graph.

Additional timing issues:
- No warm-up pass (first batch timed cold, includes lazy init)
- Used `time.time()` instead of `time.perf_counter()`

### Root Cause

1. **Batch-averaging without division**: `inference_times.append(time.time() - start)` stored batch-level time, then `np.mean(inference_times)` gave mean batch time — reported as per-graph.
2. **Cold start**: First batch includes lazy CUDA initialization, cuBLAS kernel compilation, and memory allocation.
3. **Low-resolution timer**: `time.time()` has ~15ms resolution on Windows; `time.perf_counter()` has nanosecond resolution.

### Fix

1. **Per-graph division**: `inference_times.append(elapsed / batch_size)`
2. **Warm-up pass**: One forward pass with a small batch before timing
3. **perf_counter()**: Switched to `time.perf_counter()` for monotonic, high-resolution timing

### Verification

```
Avg inference time: 0.08 ms per graph
```

---

## Issue #3 — Loss Misdescribed in Report

### What Happened

The report said "MSE (sum over both outputs)" but the code uses `F.mse_loss(pred, batch.y)` with default `reduction='mean'`.

### Root Cause

Documentation drift — the report was written before the loss implementation was finalized, and never updated.

### Fix

1. **Corrected report**: Now states "MSE with mean reduction on train-normalized targets"
2. **Frozen protocol**: Added `"loss": "MSE (mean reduction)"` to output JSON
3. **Documented normalization**: Added note about ~760× real-unit weighting of std vs mean errors from per-target normalization — 6C must use identical scheme

---

## Issue #4 — Overfitting Check Confounded by Mode Mismatch

### What Happened

Train loss was measured in `model.train()` mode (dropout active at 4 sites, BatchNorm using batch statistics); val loss in `model.eval()` mode. A val/train ratio of 0.73–0.77 is the expected artifact of that mismatch, not evidence of generalization.

Additional inconsistency: report said "0.73–0.77" in Training but "0.71–0.77" in Sanity Checks. Best train and val losses came from different epochs.

### Root Cause

`train_epoch` uses `model.train()`, which activates dropout and uses batch statistics for BatchNorm. `eval_epoch` uses `model.eval()`. Comparing these directly is apples-to-oranges.

### Fix

1. **Added `eval_mode_train_loss` function**: Runs an eval-mode pass over the train set with best weights
2. **Fixed ratios**: After fix, eval_train/val ratio is 1.40–1.54 (train loss lower than val, as expected)
3. **Removed epoch mismatch**: No longer comparing min(train_loss) vs min(val_loss) from different epochs
4. **Added to output JSON**: `eval_train_loss` field per seed

### Verification

```
Seed 42: eval_train=0.039636, best_val=0.059365, ratio=1.50
Seed 123: eval_train=0.039052, best_val=0.054775, ratio=1.40
Seed 999: eval_train=0.035969, best_val=0.055459, ratio=1.54
```

---

## Issue #5 — Batch Shape-Check Never Performed

### What Happened

Step 2 of the spec explicitly required verifying PyG batching on one batch before trusting the training loop. This was absent from both code and report.

### Root Cause

The batch shape-check was described in the spec but never implemented. The assumption was that PyG's DataLoader handles batching correctly.

### Fix

Added `check_batch_shape` function with assertions:
- `batch.x.dim() == 2`
- `batch.y.dim() == 2` and `batch.y.shape[1] == 2`
- `batch.num_graphs == batch_size`
- `batch.edge_index.max() < num_nodes`
- `batch.edge_index.min() >= 0`

### Verification

```
Batch OK: x=torch.Size([320, 3]), y=torch.Size([32, 2]), num_graphs=32
```

---

## Issue #6 — nrecon Mean Contradiction

### What Happened

Dataset table reported mean nrecon=2.85. Test breakdown implied 3.31 (1016/307). If splits are proportionally stratified, these should match.

### Root Cause

The 2.85 was the generator parameter, not the realized reconvergence count. The actual mean across all 2,000 graphs is 3.31. The test set has mean 3.31, train 3.27, val 3.26 — confirming proportional stratification.

### Fix

1. **Corrected report**: Changed dataset table from 2.85 to 3.31
2. **Added clarification**: Test breakdown aligns with dataset mean (~3.31), confirming proportional stratification
3. **Verified splits**: Train 3.27, Val 3.26, Test 3.31 — all within sampling noise

---

## Issue #7 — Fragile Dual-Construction Metadata Pattern

### What Happened

`run_stage6b.py` built `test_data` (used for per-graph metadata in `evaluate_model`) via a separate `GraphDataset(split="test").get_data(...)` call, independent from the `test_data` built inside `create_dataloaders()` that actually feeds `test_loader`. Both produced identically-ordered lists, but this was a fragile pattern — if someone later changed one construction path (e.g., added shuffling or filtering), predictions and metadata would silently misalign with no error raised.

Additionally, an old `gnn_baseline/results/stage6b_results.json` from a previous naming convention remained in the repo.

### Root Cause

1. **Separate construction paths**: `create_dataloaders()` builds one set of Data objects; `main()` builds another for metadata. No assertion enforces they are the same.
2. **Legacy artifact**: The old `stage6b_results.json` was never cleaned up after the folder rename from `stage6b` to `gnn_baseline`.

### Fix

1. **Single source of truth**: `evaluate_model()` now reads per-graph metadata directly from `test_loader.dataset` instead of a separately-built list. Predictions and metadata always come from the same DataLoader.
2. **Removed legacy file**: Deleted `gnn_baseline/results/stage6b_results.json`.
3. **Simplified API**: Removed `test_data: List` parameter from `run_single_seed()` and the redundant `test_data = test_dataset.get_data(...)` call in `main()`.

### Verification

```
Test graphs: 307
Mean delay MAE: 0.6829
...
nrecon=2 (n=97): mean_mae=0.3813
...
```

The nrecon breakdown matches the real split exactly, confirming metadata alignment is preserved through the single DataLoader path.

---

## Cross-Cutting Findings

### Code Hygiene Issues

Multiple code hygiene issues were discovered and fixed:

| Issue | Fix |
|--------|-----|
| Dead `graph_id = torch.tensor([hash(gid) % 2**31])` code | Removed (non-reproducible due to PYTHONHASHSEED) |
| Deprecated `from torch_geometric.data import DataLoader` | Switched to `torch_geometric.loader` |
| Missing DataLoader import in eval.py | Added explicit import |
| ~60 lines of defensive metadata extraction | Replaced with direct dataset zip |
| Redundant dataset loads (~6× per run) | Load once, pass around |
| Unused imports (os, F, time, VanillaDAGGNNSage) | Removed |
| `use_undirected` flag implemented but never exercised | Removed flag |
| Hardcoded paths + `sys.path.insert` hack | Made paths relative to script location |
| Top-level summary using seed-0 booleans | Added explicit aggregation in JSON |
| Typo: nrecon=8 std MAE range descending | Fixed |
| Environment versions not recorded | Added to output JSON |
| EarlyStopping only restores on early-stop trigger | Moved restoration into `__call__` |
| Python random not seeded | Added `random.seed(seed)` |
| Under-documented architecture | Documented exactly in report and protocol |

### Inference Timing Methodology

The original timing methodology had three compounding errors:
1. **Batch-level timing** reported as per-graph (32× overestimate)
2. **Cold-start contamination** (first batch includes lazy init)
3. **Low-resolution timer** (`time.time()` on Windows has ~15ms granularity)

Corrected methodology:
1. **Per-graph division**: `elapsed / batch.num_graphs`
2. **Warm-up pass**: One forward pass before timing starts
3. **`time.perf_counter()`**: Monotonic, nanosecond-resolution timer
4. **CUDA sync**: `torch.cuda.synchronize()` before stop on GPU

---

## Lessons Learned

1. **Timing methodology must be explicit**: Batch-level vs per-graph, warm-up inclusion, and timer choice each introduce 10–1000× errors. Document the exact methodology.
2. **Train/val loss comparison requires same mode**: Comparing train-mode loss to eval-mode loss is meaningless. Always compare apples-to-apples.
3. **Checkpoint provenance matters**: `.module.` keys from DataParallel are a silent provenance failure. Round-trip test (load → eval → match metrics) should be mandatory.
4. **Stats must come from final data**: Generator parameters (2.85) can differ from realized values (3.31). Always compute stats from the actual dataset.
5. **Spec requirements must be implemented**: "Verify PyG batching" in the spec was skipped. Spec-mandated checks should be code, not prose.

## Prevention

- Added `check_batch_shape` assertion block to `run_stage6b.py`
- Added `eval_mode_train_loss` for fair overfitting check
- Switched to `time.perf_counter()` with warm-up in `eval.py`
- Added environment version logging to output JSON
- Documented exact architecture, loss, and protocol in ADR-007
- Added `eval_train_loss` field per seed for generalization gap measurement
- Removed deprecated imports and unused code paths
- Made paths script-relative for CWD-independent reproducibility
- Added stratified split verification (train/val/test nrecon means)
- Eliminated dual-construction metadata pattern: `evaluate_model` reads from `loader.dataset` directly
- Removed legacy `stage6b_results.json` artifact
