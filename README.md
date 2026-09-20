# 📓 VLSI Statistical Static Timing Analysis

### The Engineering Playbook

This repository documents the complete build of a physics-informed SSTA framework, from Monte Carlo ground truth through analytical Clark MAX approximations. It is written from real implementation experience, not theory. It serves as a continuous record of how we build timing analysis tools, designed for engineers to understand the "why" behind the "what."

---

## 🧭 How to Read This

> [!IMPORTANT]  
> **Every pattern here comes from a real decision made under real constraints. If it links to a module, that's where we actually built it.**

This is not a generic list of "best practices." If a pattern or architecture choice is documented here, it means we have implemented it, validated it against Monte Carlo reference data, and dealt with its consequences in production-style code.

---

## 🗺️ Navigation

| Section | Description |
|---------|-------------|
| **[Foundations](./foundations/)** | Core theory, parameters, and frozen configuration. |
| **[Variation](./variation/)** | Process variation modeling — Monte Carlo sampler and analytical moments. |
| **[Timing](./timing/)** | Timing graph definition, DAG topology, and alpha-power delay model. |
| **[SSTA](./ssta/)** | Monte Carlo timing analysis and analytical SSTA engines. |
| **[Experiments](./experiments/)** | Orchestration scripts, reproducibility checks, and sanity tests. |
| **[Results](./results/)** | Locked reference distributions, raw data, and final reports. |
| **[Zenodo Release Kit](./zenodo/)** | Dataset release kit — pissta-2k/5k/10k + OOD-100, manifests, regeneration & verification scripts. |
| **[docs/ADRs](./docs/adrs/)** | Architecture Decision Records for key technical choices. |

---

## 🎯 Current Status

| Stage | Status | Description |
|-------|--------|-------------|
| Stage 1 | Complete | Monte Carlo baseline (feed-forward chain) |
| Stage 2 | Complete | Correlated process variation (inter-die + spatial + mismatch) |
| Stage 3 | Locked | Branching DAG reference distribution (N=100k, seed 42) |
| Stage 4 | Complete | Analytical SSTA with Clark MAX approximation |
| Stage 4b | Complete | Hybrid empirical+Clark gap decomposition |
| Stage 5 | Complete | Tail-aware SSTA with skew-normal 3-moment MAX |
| Stage 6A | Complete | Arbitrary-DAG dataset generation (2,000 graphs, MC labels via name-aligned sampling) |
| Stage 6B | Complete | Vanilla DAG-GNN baseline (beats analytical SSTA 3/3 seeds; lockstep-frozen) |
| Stage 6C | Complete | Physics-informed — feature injection added no headroom (original Tier A/A+B sig. *worse*; redesigned A+B *ns*; B-only *ns*), but architectural MAX-biased aggregation **beat vanilla capacity-matched (Δ −0.062, CI [−0.088,−0.037])** — the arc's first positive result. See `docs/adrs/ADR-008-*.md` (addendum #2) |
| Stage 7 | Complete | Split conformal calibration (studentized residual) closes the "calibrated uncertainty" half of the thesis: pooled 90% coverage **maxbias_cm 0.908 / vanilla 0.895**; eval MAE 0.371 vs 0.416. **Caveat:** both backbones under-cover nrecon=2 (MAX-heavy reconvergence regime); no coverage claim under distribution shift. See `docs/adrs/ADR-009-*.md` |
| Stage 9 | Complete | OOD-100 cross-method comparison — under the distribution shift that collapses coverage (25–27%), GNN point error degrades ~6× (2.38–2.57 MAE) vs the validated analytical baseline's 1.9× (0.2111); paired cluster-bootstrap CIs exclude 0, analytical wins 99–100% of graphs. See `results/stage9_ood_crossmethod_report.md` |
| Stage 10 | Complete | pissta-10k scale test — fresh 10k training (0.289) and zero-shot (0.42) both remain significantly behind the analytical baseline (0.1101, 0.50 ms/graph); the "may erode at scale" hypothesis is rejected; MAX-bias advantage washes out at 10k (ns). See `results/stage10_scale_test_report.md` |
| Stage 11 | Complete | Stage 4 fixed-topology audit — D1/D2 defect patterns absent (720-permutation readback invariance; Pelgrom cap inactive), 0.2781 gap decomposition reproduced bit-exactly (pre-fix audit figure 0.3083; re-run CLEAN after the 2026-09-20 propagate_path covariance fix); §IV pipeline independently verified. See `results/stage11_stage4_audit_report.md` |

---

## 🏗️ Canonical Layout

```text
project-root/
├── LICENSE                    # MIT (code) + CC-BY-4.0 (data)
├── foundations/               # Theory, configs, frozen parameters
│   ├── README.md
│   ├── config_loader.py
│   ├── stage3_config.json
│   └── stage3_config_asymmetric.json
├── variation/                 # Process variation modeling
│   ├── README.md
│   ├── __init__.py
│   ├── sampler.py             # MC variation sampler (inter-die + spatial + Pelgrom)
│   └── analytical.py          # Closed-form moments (mean/var/cov)
├── timing/                    # Timing graph and delay models
│   ├── README.md
│   ├── __init__.py
│   ├── graph.py               # DAG topology, topological sort
│   ├── delay.py               # Alpha-power delay with geometry sensitivity
│   ├── clark_max.py           # Clark's Gaussian MAX approximation
│   └── tail_aware_max.py      # Skew-normal 3-moment MAX approximation
├── ssta/                      # SSTA analysis engines
│   ├── README.md
│   ├── __init__.py
│   ├── monte_carlo.py         # Vectorized MC timing analysis
│   ├── analytical_ssta.py     # Full analytical pipeline
│   ├── statistical_sum.py     # Gaussian sum propagation
│   └── statistics.py          # Summary statistics utilities
├── experiments/               # Orchestration and validation
│   ├── README.md
│   ├── __init__.py
│   ├── run_stage3.py          # Reference MC run (N=100k)
│   ├── run_stage4b_hybrid.py  # Empirical+Clark gap decomposition
│   ├── run_stage5.py          # Tail-aware skew-normal SSTA
│   ├── reproducibility_check.py
│   ├── asymmetric_sanity_check.py
│   └── sanity_checks.py
├── results/                   # Locked outputs and reports
│   ├── README.md
│   ├── stage3_arrival_times.npy
│   ├── stage3_cpd.npy
│   ├── stage3_raw.npz
│   ├── stage3_summary.json
│   ├── stage4_analytical_ssta.json
│   ├── stage4b_hybrid_run.json
│   ├── stage5_tail_aware_ssta.json
│   └── ...
├── docs/
│   └── adrs/                  # Architecture Decision Records
├── data_generation/           # Stage 6A training data generation
│   ├── README.md
│   ├── run_stage6a.py         # DAG dataset generator
│   ├── graph_generator.py     # Random valid DAG generator
│   ├── analytical_ssta_arbitrary.py  # Arbitrary-DAG analytical SSTA
│   └── data/
├── zenodo/                    # Zenodo dataset release kit (see zenodo/README.md)
│   ├── CODE_README.md         # Regeneration + verification + upload guide
│   ├── scripts/               # generate_pissta.py, verify_release.py
│   └── data/                  # generated releases (gitignored)
├── gnn_baseline/              # Stage 6B/6C GNN surrogate modeling
│   ├── run_stage6b.py         # Vanilla DAG-GNN baseline
│   ├── run_stage6c.py         # Physics-informed ablation
│   ├── model.py               # GraphSAGE + physics-informed models
│   ├── train.py               # Training loop + EarlyStopping
│   ├── eval.py                # Metrics + analytical comparison
│   ├── dataset.py             # PyG Data loaders
│   └── results/
├── tests/                       # Unit tests (pytest)
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## 📄 License

Code: **MIT** · Data (Zenodo deposits): **CC-BY-4.0** — see [LICENSE](./LICENSE).

---

_See [CHANGELOG.md](./CHANGELOG.md) for recent updates._
