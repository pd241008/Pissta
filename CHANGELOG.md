# 📜 Changelog

> All notable changes to this project will be documented in this file.

---

### August 21, 2026 — Bug-Fix Sweep + Verified Stage 6B/6C Re-run

- 🐛 **Fixed MC column misalignment** (`ssta/monte_carlo.py`): sample columns (ordered by `gate_coords`) were indexed by *topological* position, so arbitrary Stage 6A DAGs received other gates' variation samples; also made labels depend on process-level set-iteration order. Now mapped by NAME. Verified no-op for the locked Stage 3 reference (exact match to 4 decimals).
- 🐛 **Fixed physics timing measurement** (`gnn_baseline/run_stage6c.py`): default-config `variation_params` (G1–G6) vs generated gate names caused `compute_analytical_ssta_arbitrary` to raise KeyError, silently swallowed — tier_ab timing never measured the analytical SSTA. Now builds per-graph variation params; silent except removed.
- 🐛 **Fixed CUDA non-determinism** (both runners): `set_seed` now calls `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Previously 6B-vs-6C lockstep could not reproduce bit-exact on GPU.
- 🐛 **Fixed latent NameError** in `run_branching_monte_carlo` default-graph path (`build_branching_graph` was never imported) and CWD-dependent default path in `load_config` / `run_stage6a.py`.
- 🧪 **Wired up dead `preflight()`** validation in Stage 6C runner.
- 🔁 **Re-ran end-to-end**: regenerated 6A dataset (2000 graphs), re-trained 6B and full 6C 3-way ablation. Lockstep verification now exact (max diff 0.0).
- ⚠️ **Revised conclusions**: on corrected labels, Tier A (+0.040 MAE, CI [+0.024,+0.056]) and Tier A+B (+0.063, CI [+0.015,+0.112]) are significantly **worse** than vanilla (0.408 ± 0.006). The previously reported "Tier A+B −21%" improvement does not reproduce — it was an artifact of the misaligned-label dataset and its different graph-complexity mix. Vanilla GNN remains the best model (analytical baseline: 0.661).

### August 16, 2026 — Stage 5 Tail-Aware SSTA (reporting tightened)

- ✨ **Implemented** skew-normal 3-moment MAX approximation (`timing/tail_aware_max.py`)
- ✨ **Implemented** Stage 5 orchestration (`experiments/run_stage5.py`)
- 📊 **Results:** Tail-aware P99.87 = 10.6584 vs pooled MC P99.87 = 10.7475 (gap = 0.0891)
- 📉 **Gap closure:** 56.9% of shape gap vs pooled MC reference (0.1192 / 0.2083)
- 🔢 **Absolute gaps:** 0.0952 / 0.0783 / 0.0938 across seeds 42/123/999 (mean 0.0891 ± 0.0090)
- ⚡ **Runtime:** ~140 ms — ~1.5× faster than full MC, ~350× slower than pure Clark
- 🛡️ **Noise-corrected reporting:** closure% reported vs pooled MC only; seed-to-seed swing quantified as MC reference noise

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
