# results

Locked reference distributions, raw data, and final reports.

## Contents

### Stage 9–11 (Cross-Method Follow-ups: OOD accuracy, 10k scale test, Stage 4 audit)

| File | Description |
|------|-------------|
| `stage9_ood_crossmethod_report.md` | OOD-100 cross-method comparison: GNN per-graph MAE (2.38–2.57) vs validated analytical (0.2111) on the 100 OOD graphs, paired cluster-bootstrap CIs, nrecon breakdown — the point-accuracy counterpart to Stage 7 §6 coverage collapse |
| `stage10_scale_test_report.md` | pissta-10k scale test: fresh 10k training (0.289–0.295) vs zero-shot (0.42–0.43) vs analytical (0.1101, 0.50 ms/graph), paired bootstraps, "may erode at scale" hypothesis rejected, MAX-bias advantage washes out at 10k (ns) |
| `stage11_stage4_audit_report.md` | Stage 4 fixed-topology audit: D1/D2 defect patterns absent (720-permutation invariance; Pelgrom cap inactive), 0.3083 gap decomposition reproduced bit-exactly — confirmatory note closing the scope caveat |

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

### Stage 7 (Conformal Calibration) & Stage 8 (Cross-Method Comparison)

| File | Description |
|------|-------------|
| `stage7_report.md` | Split-conformal calibration report (ADR-009), incl. nrecon=2 under-coverage caveat |
| `stage8_report.md` | **Final cross-method cost/accuracy table** (MC / analytical / tail-aware / vanilla / MAX-bias CM + full-cap / combined × mean-MAE · inference ms · training s · amortized data-gen) with honest amortized-cost paragraph |

### Stage 6C (Physics Feature Injection + MAX-bias artifact set)

| File | Description |
|------|-------------|
| `stage6c_report.md` | Comprehensive Stage 6C cross-lever report (all four feature attempts + MAX-bias arc) |
| `stage6c_results.json` | **≥0.365/0.367/0.361** (maxbias_cm 7-seed, maxbias full-cap, combined tierab) lockstep-verified runs w/ training-time instrumentation |
| `stage6c_results_maxbias_cm_tierab.json` | Combined MAX-bias-CM × Tier A+B ablation (7 seeds) — first execution |
| `stage6c_results_original_design.json` | Stage 6C original-design runs (pre-hardening, archived) |

### Stage 7 (Split-Conformal Calibration) & Stage 8 (Cross-Method Comparison)

| File | Description |
|------|-------------|
| `stage7_report.md` | Split-conformal calibration report (ADR-009) with the nrecon=2 under-coverage caveat |
| `stage8_report.md` | **Final cross-method cost/accuracy table** — MC / analytical (Clark) / tail-aware skew-normal / vanilla / MAX-bias (CM + full-cap) / combined × mean-MAE, inference ms/graph, training s, amortized data-gen — with honest amortized-cost paragraph |

### Stage 6C (Physics Feature Injection / MAX-bias arc)

| File | Description |
|------|-------------|
| `stage6c_report.md` | Cross-lever Stage 6C report (all four feature attempts + MAX-bias arc) |
| `stage6c_results.json` | Committed artifact: vanilla/maxbias_cm (7 seeds) w/ train_time instrumentation |
| `stage6c_results_maxbias.json` | MAX-bias full-capability (3 seeds) w/ train_time instrumentation |
| `stage6c_results_maxbias_cm_tierab.json` | Combined MAX-bias-CM + redesigned Tier A+B ablation (7 seeds) — first execution |
| `stage6c_results_nocoor.json` | Vanilla-minus-coordinates diagnostic (3 seeds) |
| `stage6c_results_tier_b_only.json` | Tier-B-only ablation (3 seeds) |
| `stage6c_results_original_design.json` | Original-design feature-injection runs (archived, pre-hardening) |

Pre-2026-09-13 backup of the pre-train-time versions: `gnn_baseline/results_backup_pre-traintime/`, `gnn_baseline/checkpoints_backup_pre-traintime/`; a second copy in `/tmp/opencode/artifacts_backup/`.

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
