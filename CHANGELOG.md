# 📜 Changelog

> All notable changes to this project will be documented in this file.

---

### August 16, 2026 — Stage 5 Tail-Aware SSTA

- ✨ **Implemented** skew-normal 3-moment MAX approximation (`timing/tail_aware_max.py`)
- ✨ **Implemented** Stage 5 orchestration (`experiments/run_stage5.py`)
- 📊 **Results:** Tail-aware P99.87 = 10.6584 vs MC P99.87 = 10.7536 (gap = 0.0952)
- 📉 **Gap closure:** 54.28% of shape gap (0.1131 / 0.2083) — within expected 40–120% range
- ⚡ **Runtime:** 137 ms — ~1.5× faster than full MC, ~350× slower than pure Clark
- 🌐 **Multi-seed validation** — closure consistent across seeds 42, 123, 999

### August 16, 2026 — Monorepo Restructure

- 🏗️ **Reorganized** repository into modular monorepo structure per Design Dungeons standards
- 📁 **Created** modular directories: `foundations/`, `variation/`, `timing/`, `ssta/`, `experiments/`, `results/`
- 📖 **Added** module-level READMEs with descriptions and interfaces
- 📝 **Created** `docs/adrs/` for Architecture Decision Records
- 📋 **Created** `docs/postmortems/` for lessons learned
- 🧪 **Created** `tests/` directory for unit tests
- 📊 **Created** `CHANGELOG.md` and updated root `README.md`

### August 15, 2026 — Stage 4 Analytical SSTA

- ✨ **Implemented** analytical variation moments (closed-form mean/var/cov)
- ✨ **Implemented** linearized alpha-power delay model
- ✨ **Implemented** Clark MAX approximation for reconvergent paths
- ✨ **Implemented** full analytical SSTA pipeline with Stage 3 comparison
- 📊 **Results:** Analytical P99.87 = 10.4454 vs MC P99.87 = 10.7536 (gap = 0.3083)
- ⚡ **Speedup:** 566× faster than N=100k Monte Carlo (0.35 ms vs 200 ms)
- 🔍 **Validated** covariance bookkeeping — analytical covariances match empirical within 3–5%
- 🌐 **Multi-seed validation** — error consistent across seeds 42, 123, 999

### August 15, 2026 — Stage 3 Reference Lock

- 🔒 **Locked** N=100k reference distribution (seed 42)
- 📊 **Headline stats:** mean=9.3065, std=0.4194, P95=10.0243, P99=10.3675, P99.87=10.7536
- ✅ **Asymmetric sanity check** passes — extra gate G7 makes Path 2 critical 100% of the time
- 💾 **Persisted** raw .npy arrays and compressed .npz archive
- 🔁 **Reproducibility** validated across 3 seeds — delta P99.87 stable at ~0.325 ± 0.012

### August 14, 2026 — Stage 1 Baseline

- ✨ **Implemented** Monte Carlo ground truth for 5-gate feed-forward chain
- 📊 **Baseline stats:** mean=13.6251, std=0.8824, P95=15.1103, P99=15.7961, P99.87=16.6037
