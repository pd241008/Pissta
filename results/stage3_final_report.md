# Stage 3 Final Report — Branching Timing Graph under Correlated Process Variation

**Date:** 2026-08-15  
**Stage:** Stage 3 — Monte Carlo Reference Distribution  
**Status:** Sign-off complete (N=100k, seed 42 locked as ground truth)

---

## 1. DAG Structure

```
              ┌── G2 ── G4 ──┐
Input → G1 ──┤                ├── G6 → Output
              └── G3 ── G5 ──┘
```

- **Path 1 (via G2/G4):** G1 → G2 → G4 → G6  
- **Path 2 (via G3/G5):** G1 → G3 → G5 → G6  
- **Reconvergence:** Both paths merge at G6 (sink).  
- **Gate coordinates (for spatial covariance):**
  - G1: (0.0, 0.0) — input
  - G2: (0.0, 2.0) — upper branch
  - G3: (0.0, -2.0) — lower branch
  - G4: (2.0, 2.0) — upper branch
  - G5: (2.0, -2.0) — lower branch
  - G6: (4.0, 0.0) — output/merge

---

## 2. Configuration Summary

All parameters are frozen in `stage3_config.json`. Key values:

### 2.1 Variation Parameters
| Parameter | Value | Units |
|-----------|-------|-------|
| Nominal L | 45.0 | nm |
| Nominal W | 90.0 | nm |
| Nominal Vth | 0.40 | V |
| Inter-die σ_L | 0.5 | nm |
| Inter-die σ_W | 1.0 | nm |
| Inter-die σ_Vth | 0.005 | V |
| Spatial σ_L | 1.0 | nm |
| Spatial σ_W | 1.5 | nm |
| Spatial σ_Vth | 0.01 | V |
| Spatial correlation length λ | 3.0 | (normalized units) |
| Random σ_L | 0.5 | nm |
| Random σ_W | 1.0 | nm |
| Pelgrom A_Vth | 1e-3 | V·μm |
| Pelgrom S_Vth | 1e-2 | V/μm |

### 2.2 Timing Parameters (Alpha-Power Model)
| Parameter | Value |
|-----------|-------|
| k | 1.0 |
| α | 1.3 |
| Vdd | 1.0 | V |
| Gate loads (C_eff proxy): G1=1.0, G2=1.1, G3=1.1, G4=1.2, G5=1.2, G6=1.4 |

### 2.3 Experiment Settings
| Setting | Value |
|---------|-------|
| N_samples (reference) | 100,000 |
| Reference seed | 42 |
| Reproducibility seeds | [42, 123, 999] |
| Output directory | results/ |

---

## 3. Reference Run Results (N=100,000, Seed 42)

These are the **locked ground-truth numbers** for Stage 4+ comparison.

### 3.1 Critical-Path Delay Distribution (Branching DAG)
| Metric | Value |
|--------|-------|
| Mean | 9.3065 |
| Std | 0.4194 |
| P95 | 10.0243 |
| P99 | 10.3675 |
| P99.87 | **10.7536** |
| Min | 7.6830 |
| Max | 11.7000 |

### 3.2 Critical-Path Split
| Branch | Fraction Critical |
|--------|-------------------|
| Path via G4 | 49.94% |
| Path via G5 | 50.06% |

Both paths are genuinely competitive; no structural domination.

### 3.3 Raw Data Files Persisted
| File | Description | Size |
|------|-------------|------|
| `results/stage3_cpd.npy` | Critical-path delay array (100k floats) | ~800 KB |
| `results/stage3_path_G4_critical.npy` | Per-sample branch labels (G4 side) | ~100 KB |
| `results/stage3_path_G5_critical.npy` | Per-sample branch labels (G5 side) | ~100 KB |
| `results/stage3_L_nm.npy` | Per-gate L draws (N, 6) | ~4.8 MB |
| `results/stage3_W_nm.npy` | Per-gate W draws (N, 6) | ~4.8 MB |
| `results/stage3_Vth_v.npy` | Per-gate Vth draws (N, 6) | ~4.8 MB |
| `results/stage3_arrival_times.npy` | Full AT matrix (N, 6) | ~4.8 MB |
| `results/stage3_raw.npz` | Compressed archive of all above | ~22.9 MB |

---

## 4. Reproducibility & Stability (N=100,000 across Seeds)

Multi-seed validation confirms the MAX-induced gap is not a fluke.

### 4.1 Per-Seed Results
| Seed | Branching Mean | Branching Std | Branching P99.87 | Chain P99.87 | **Delta P99.87** | Gaussian Approx (mean+3σ) |
|------|----------------|---------------|------------------|--------------|------------------|---------------------------|
| 42   | 9.3065 | 0.4194 | 10.7536 | 10.4406 | **0.3131** | 10.5648 |
| 123  | 9.3076 | 0.4185 | 10.7367 | 10.4103 | **0.3263** | 10.5632 |
| 999  | 9.3071 | 0.4201 | 10.7522 | 10.4151 | **0.3371** | 10.5674 |

### 4.2 Stability Summary
| Statistic | Value |
|-----------|-------|
| Mean delta P99.87 across seeds | 0.3255 |
| Std of deltas across seeds | 0.0120 |
| Path G4 critical fraction (range) | 49.7% – 50.2% |
| Path G5 critical fraction (range) | 49.8% – 50.3% |

**Conclusion:** The gap is stable and reproducible. A std of 0.012 on a ~0.32 effect is a 27:1 signal-to-noise ratio.

---

## 5. Sanity Checks

### 5.1 Chain Comparison (Same Variation Draws)
To isolate the effect of the MAX operation, we ran a simple 4-gate chain through the **same** variation samples. The chain ignores branches and simply sums delays.

| Metric | Branching DAG | Chain (same variation) | Delta |
|--------|--------------|------------------------|-------|
| Mean | 9.3065 | 9.1623 | +0.1442 |
| Std | 0.4194 | 0.3842 | +0.0352 |
| P95 | 10.0243 | 9.8136 | +0.2107 |
| P99 | 10.3675 | 10.1144 | +0.2531 |
| P99.87 | 10.7536 | 10.4406 | **+0.3131** |

**Key observation:** The branching DAG has a **higher mean** and **heavier tail** than the chain, even though both use identical per-gate variation. This is the signature of the MAX operation: when both paths are delayed, the worst path wins, pushing the tail upward.

### 5.2 Gaussian Approximation Failure
| Method | P99.87 Estimate | Actual P99.87 | Error |
|--------|-----------------|---------------|-------|
| Gaussian (mean + 3σ) | 10.5648 | 10.7536 | **−0.1888** |

The Gaussian approximation underestimates the tail by ~0.19 (1.8% relative error on the delay value). This confirms that the MAX of two correlated delays is **non-Gaussian** and has a heavier tail than a simple sum.

### 5.3 Asymmetric Sanity Check (AT/Argmax Validation)
We modified the DAG to make Path 2 structurally longer by adding an extra gate (G7):

```
G1 → G3 → G5 → G7 → G6   (5 gates)
G1 → G2 → G4 → G6         (4 gates)
```

| Metric | Value |
|--------|-------|
| Mean | 11.7263 |
| Std | 0.5499 |
| P95 | 12.6638 |
| P99 | 13.1007 |
| P99.87 | 13.6048 |
| Path via G4 critical | 0.00% |
| Path via G7 critical | **100.00%** |

**PASS:** The longer path is critical in every sample. This confirms the AT propagation and MAX argmax logic are correct. If the split had remained near 50/50, it would have indicated a bug.

---

## 6. Tail Sensitivity: 10k vs 100k

The 10k diagnostic run (`experiments/sanity_checks.py`) shows tail sensitivity:

| Metric | N=10,000 (diagnostic) | N=100,000 (reference) |
|--------|----------------------|----------------------|
| Branching P99.87 | 10.6803 | 10.7536 |
| Chain P99.87 | 10.4227 | 10.4406 |
| Delta P99.87 | 0.2576 | 0.3131 |
| Gaussian approx | 10.5706 | 10.5648 |

**Note:** The 10k delta is noisier (0.258 vs 0.313). **Do not use 10k numbers as ground truth.** The N=100k reference is the locked distribution.

---

## 7. Critical-Path Fraction Diagnostic

For the balanced DAG, the critical-path split is ~50/50, which is essential for exercising the MAX behavior:

| DAG | Path G4 Critical | Path G5 Critical |
|-----|------------------|------------------|
| Balanced (reference) | 49.9% | 50.1% |
| Asymmetric (G7 added) | 0.0% | 100.0% |

If the split were 100/0 in the balanced DAG, the two paths would not be balanced enough to make the MAX problem interesting.

---

## 8. Source Code Inventory

| File | Purpose |
|------|---------|
| `stage3_config.json` | Canonical config: DAG, variation, timing, experiment |
| `config_loader.py` | JSON config loader and validator |
| `variation.py` | Correlated L/W/Vth sampler (inter-die + spatial PCA + Pelgrom) |
| `timing_graph.py` | DAG topology, topological sort, alpha-power delay |
| `monte_carlo.py` | Vectorized AT propagation, MAX critical-path selection |
| `statistics.py` | Mean/std/P95/P99/P99.87, path-split utility |
| `experiments/run_stage3.py` | Reference orchestration (N=100k, seed 42) |
| `experiments/reproducibility_check.py` | Multi-seed stability validation |
| `experiments/asymmetric_sanity_check.py` | AT/argmax logic validation |
| `experiments/sanity_checks.py` | Diagnostic N=10k comparison |

---

## 9. Important Disclaimers

1. **Normalized / toy units.** All numerical values (α = 1.3, Vdd = 1.0, 45 nm nominal L, etc.) are deliberately simple demonstration parameters for software validation. They are **NOT** silicon-calibrated and should **NOT** be used in any publication claiming technology-specific experimental data.

2. **No analytical SSTA yet.** This stage is pure Monte Carlo. The reference distribution is the ground truth against which Stage 4 (analytical SSTA), Stage 5 (MAX approximations like Clark), and later stages (GNN surrogates, conformal calibration) will be graded.

3. **10k is diagnostic-only.** The N=10,000 runs are useful for quick sanity checks but have observable tail noise. The N=100,000 reference is the locked ground truth.

---

## 10. Sign-Off Checklist

- [x] N=100k reference run completed (seed 42)
- [x] Raw data persisted (.npy + .npz)
- [x] Reproducibility validated across 3 seeds (delta P99.87 stable at ~0.325 ± 0.012)
- [x] Chain comparison confirms MAX-induced gap (branching > chain)
- [x] Gaussian approximation shown to underestimate tail
- [x] Asymmetric DAG confirms AT/argmax logic (100% path selection)
- [x] Critical-path split ~50/50 in balanced DAG
- [x] Config frozen in JSON (no hardcoded parameters)
- [x] README updated with all headline stats and disclaimers

**Stage 3 is complete and ready for Stage 4.**
