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
| **[docs/ADRs](./docs/adrs/)** | Architecture Decision Records for key technical choices. |
| **[docs/Postmortems](./docs/postmortems/)** | Lessons learned from bugs and validation failures. |

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

---

## 🏗️ Canonical Layout

```text
project-root/
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
│   ├── adrs/                  # Architecture Decision Records
│   └── postmortems/           # Lessons learned
├── data_generation/           # Stage 6A training data generation
│   ├── README.md
│   ├── run_stage6a.py         # DAG dataset generator
│   ├── graph_generator.py     # Random valid DAG generator
│   ├── analytical_ssta_arbitrary.py  # Arbitrary-DAG analytical SSTA
│   └── data/
├── gnn_baseline/              # Stage 6B/6C GNN surrogate modeling
│   ├── run_stage6b.py         # Vanilla DAG-GNN baseline
│   ├── run_stage6c.py         # Physics-informed ablation
│   ├── model.py               # GraphSAGE + physics-informed models
│   ├── train.py               # Training loop + EarlyStopping
│   ├── eval.py                # Metrics + analytical comparison
│   ├── dataset.py             # PyG Data loaders
│   └── results/
├── tests/                       # Unit tests (pytest, 11 tests)
├── CHANGELOG.md
└── README.md
```

---

_See [CHANGELOG.md](./CHANGELOG.md) for recent updates._
