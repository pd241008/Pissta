# results

Locked reference distributions, raw data, and final reports.

## Contents

### Stage 3 (Monte Carlo Reference)

| File | Description |
|------|-------------|
| `stage3_summary.json` | Headline stats: mean/std/P95/P99/P99.87, critical-path split |
| `stage3_raw.npz` | Compressed archive of all raw arrays |
| `stage3_cpd.npy` | Critical-path delay array (100,000 floats) |
| `stage3_L_nm.npy` | Per-gate L draws (N, 6) |
| `stage3_W_nm.npy` | Per-gate W draws (N, 6) |
| `stage3_Vth_v.npy` | Per-gate Vth draws (N, 6) |
| `stage3_arrival_times.npy` | Full AT matrix (N, 6) |
| `stage3_path_G4_critical.npy` | Per-sample branch labels (G4 side) |
| `stage3_path_G5_critical.npy` | Per-sample branch labels (G5 side) |

### Stage 4 (Analytical SSTA)

| File | Description |
|------|-------------|
| `stage4_analytical_ssta.json` | Full analytical report with comparison vs MC |
| `stage4_final_report.md` | Human-readable Stage 4 report |

### Stage 4b (Hybrid Run)

| File | Description |
|------|-------------|
| `stage4b_hybrid_run.json` | Empirical+Clark gap decomposition |

### Stage 5 (Tail-Aware SSTA)

| File | Description |
|------|-------------|
| `stage5_tail_aware_ssta.json` | Tail-aware SSTA report with skew-normal MAX, gap closure analysis |

### Diagnostic/Validation

| File | Description |
|------|-------------|
| `stage3_reproducibility_N100k.json` | Multi-seed stability results |
| `stage3_asymmetric_check.json` | Asymmetric DAG sanity check |
| `stage3_sanity_checks.json` | N=10k diagnostic comparison |
| `stage3_histogram.png` | CPD histogram visualization |

## Ground Truth Lock

- **Reference seed:** 42
- **N_samples:** 100,000
- **Locked:** August 15, 2026

All Stage 4+ comparisons must use these exact arrays. Do not regenerate from a different seed.
