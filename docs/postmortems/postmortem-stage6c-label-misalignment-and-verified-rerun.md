# Session Report — Bug Sweep & Verified Pipeline Re-run

> **Date:** August 21, 2026
> **Repo:** Pissta (VLSI Statistical Static Timing Analysis), branch `main`
> **Starting point:** Unverified post-merge state from commits `d72f7b7` ("fixing bugs… in progress") and `4453b71` ("i give up look into this shit later")
> **Environment:** Python 3.14.6 · torch 2.9.1 · numpy 2.4.0 · torch-geometric 2.8.0.post1 (installed via `pip install --break-system-packages`) · RTX 4050 Laptop GPU · 10 cores
> **Relocated** from the repo root (`SESSION_REPORT_2026-08-21.md`) on August 21, 2026. Successor document: `postmortem-stage6c-runner-hardening.md`.

---

## 1. Objective

Audit the repo for outstanding issues, full-scan for bugs, fix them, re-run the affected pipeline stages, and confirm which bugs were fixed vs. remain.

---

## 2. Bugs Found & Fixed (6)

### B1 — Monte Carlo label misalignment *(critical)*
**File:** `ssta/monte_carlo.py`
Variation sample columns were ordered by `gate_coords` key order but indexed by *topological* position (`l_idx`). For arbitrary Stage 6A DAGs (where topological order ≠ `gate_coords` order), gates received **other gates' variation samples**, corrupting every MC label. Additionally, iteration over unordered sets made labels depend on process-level hash seeds.
**Fix:** Name-based `col_idx` mapping + `KeyError` guard for missing gates.
**Validation:** Locked Stage 3 reference (feed-forward chain, N=100k, seed 42) reproduces **exactly**: mean=9.3065, std=0.4194, P99.87=10.7536 — the fix is provably a no-op where order already matched.

### B2 — Latent NameError
**File:** `ssta/monte_carlo.py`
`run_branching_monte_carlo` called `build_branching_graph()` on its default-graph path without importing it.
**Fix:** Import added.

### B3 — Physics feature timing never measured
**File:** `gnn_baseline/run_stage6c.py` (`measure_physics_feature_time`)
Passed default-config `variation_params` (gates G1–G6) while graphs use generated names (n0, g2, …). `compute_analytical_ssta_arbitrary` raised `KeyError`, silently swallowed by `except KeyError: pass` — tier_ab's reported physics time never included the actual analytical SSTA.
**Fix:** Per-graph params via `dataclasses.replace(variation_params, gate_coords=...)`; silent except removed; dead `compute_process_moments` import/local removed.

### B4 — CUDA non-determinism *(root cause of the lockstep failure)*
**Files:** `gnn_baseline/run_stage6b.py`, `gnn_baseline/run_stage6c.py`
`set_seed()` set only `cudnn.deterministic=True` / `benchmark=False`. Training actually ran on GPU, where scatter-add atomics are non-deterministic regardless — so 6B-vs-6C lockstep could never reproduce bit-exact across processes.
**Fix:** `torch.use_deterministic_algorithms(True)` + `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")`.

### B5 — CWD-dependent paths
**Files:** `foundations/config_loader.py`, `data_generation/run_stage6a.py`
`load_config()` default `"stage3_config.json"` only resolved if CWD == foundations/; Stage 6A paths broke when run outside repo root.
**Fix:** Anchored to module location / `REPO_ROOT`.

### B6 — Dead validation code
**File:** `gnn_baseline/run_stage6c.py`
`preflight()` was defined but never called.
**Fix:** Wired into `main()` per configuration (exits on errors).

---

## 3. The Lockstep Investigation (how B4 was found)

Symptom: 6B vanilla vs 6C vanilla per-seed metrics disagreed despite identical protocol, seed, and data.

Evidence chain:
1. Training verified bit-reproducible across *back-to-back* processes (CPU probes).
2. 6B vs 6C histories: train_loss identical through epoch 2 (0.55878225), val_loss diverged at epoch 0 (~5e-6).
3. Faithful CPU replication produced a *third* trajectory (ep0 = 0.54012578).
4. Thread-count / allocator-perturbation probes: all still 0.54012578 → not threading.
5. Re-running the actual script failed to reproduce itself (best_val 0.031576 → 0.031662) → something environmental.
6. A debug copy of the script printed **`Device: cuda`** — the real runs trained on GPU; all probes had hardcoded CPU.
7. CUDA + deterministic-algorithm flags reproduced the real runs' ep0 **exactly** (0.55878225) and became bit-stable across processes.

Conclusion: GPU atomics, not code logic, caused the mismatch. Fix B4 resolves it.

---

## 4. Re-runs & Final Numbers

### Stage 6A — dataset regenerated
2,000 graphs in 102.5 s · splits 1398/298/304 · nrecon distribution {1: 693, 2: 594, 3: 109, 4: 2}.
(`dataset.pkl` md5 `aa6e8edabbeb45322d575c46e9d0d1e3`; `splits.json` md5 `585cdcdf2f0bb6aca45ca92bb9d818e8`)
Note: differs from the old dataset (old nrecon max 6) because the generator was reworked before the old data was deleted — old report numbers are not reproducible by design.

### Stage 6B — vanilla DAG-GNN (deterministic re-run)
| Seed | Mean delay MAE |
|------|----------------|
| 42   | 0.4116 |
| 123  | 0.4018 |
| 999  | 0.4111 |
| **mean** | **0.4082 ± 0.0055** |

Beats analytical SSTA baseline (MAE 0.6609) on 3/3 seeds. Saved to `gnn_baseline/results/vanilla_dag_gnn_results.json`.

### Stage 6C — 3-way ablation (deterministic re-run)
| Model | Mean MAE | Δ vs vanilla | 95% CI | Significant? |
|-------|----------|--------------|--------|--------------|
| Vanilla (6B) | **0.408 ± 0.006** | — | — | — |
| Tier A (node sensitivities) | 0.448 ± 0.022 | +0.040 | [+0.024, +0.056] | Yes — **worse** |
| Tier A+B (+ graph-level analytical) | 0.471 ± 0.010 | +0.063 | [+0.015, +0.112] | Yes — **worse** |

Supporting checks: lockstep verification **exact** (max diff 0.0, all seeds) · ResidualMLP 0.6553 ± 0.0031 vs OLS floor 0.6494 · Tier-B leakage corr 0.9721 · Tier-A corr(vth, load_ff) = 1.0000 (redundancy mechanism confirmed) · physics timing now real: tier_a ≈ 0.01 ms/graph, tier_ab ≈ 0.67 ms/graph.

---

## 5. Revised Scientific Conclusion

> On corrected labels, **neither physics tier helps — both are significantly worse than vanilla.**
> The previous headline claim ("Tier A+B reduces mean MAE by 21%, significant") **does not reproduce**. It was an artifact of the mislabeled dataset (B1) plus its different graph-complexity mix. The redundancy diagnosis for Tier A stands (r = 1.00 with load_ff); Tier B's graph-level feature remains 97% correlated with the label.

---

## 6. Documentation Updated

| File | Change |
|------|--------|
| `CHANGELOG.md` | New entry documenting all fixes + revised conclusions |
| `results/stage6c_report.md` | SUPERSEDED banner with corrected numbers; body kept as historical record |
| `README.md` | Status table extended with Stages 6A/6B/6C; layout fixes |
| `tests/README.md` | Removed stale "planned but not yet implemented" note (11 tests pass) |
| `gnn_baseline/run_stage6c.py` | Status string updated to verified-complete |

Tests: `python -m pytest tests/ -q` → **11 passed**.

---

## 7. Remaining Known Issues (documented, unfixed)

1. **Pelgrom d-cap inconsistency:** `variation/sampler.py` caps effective dimension at 5.0; `variation/analytical.py` does not. No practical impact here (max d ≈ 4.12 < 5), but formulas can diverge for large devices.
2. `data_generation/run_stage6a.py` uses `List`/`Dict` annotations without importing them — runtime-safe due to `from __future__ import annotations`.
3. All changes are **uncommitted** (9 modified files).

---

## 8. Files Changed

```
M CHANGELOG.md
M README.md
M data_generation/run_stage6a.py
M foundations/config_loader.py
M gnn_baseline/run_stage6b.py
M gnn_baseline/run_stage6c.py
M results/stage6c_report.md
M ssta/monte_carlo.py
M tests/README.md
```

Regenerated artifacts: `data_generation/data/{dataset.pkl,splits.json}`, `gnn_baseline/results/{vanilla_dag_gnn_results,stage6c_results,stage6c_results_interim}.json`, `gnn_baseline/checkpoints/*`.

## 9. Reproduce

```bash
python -m pytest tests/ -q                                   # 11 passed
python experiments/run_stage3.py                             # locked ref: 9.3065 / 0.4194 / 10.7536
python data_generation/run_stage6a.py                        # regenerate dataset
python gnn_baseline/run_stage6b.py                           # vanilla baseline
python gnn_baseline/run_stage6c.py                           # full ablation + lockstep check
```
