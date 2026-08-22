# 📜 Changelog

> All notable changes to this project will be documented in this file.

---

### August 21, 2026 — Stage 6C Runner Hardening (audit follow-up)

- 🛡️ **Status string now derived from computed blocks** (`gnn_baseline/run_stage6c.py`): the hardcoded "complete — verified end-to-end…" claim (which had been edited into the source after the verified re-run, leaving artifact/source vintages out of sync) is replaced by a status assembled at runtime from the lockstep result, CI computation, physics timing, S2 verdict, convergence gate, and B5 persistence.
- 🛡️ **Smoke/full artifacts separated by construction**: in `VLSI_SMOKE=1` mode the runner writes `stage6c_results_smoke.json` / `stage6c_results_interim_smoke.json` and checkpoints into `checkpoints_smoke/`, and tags the status `SMOKE run | …`. A smoke run can no longer overwrite full-run artifacts or inherit their claims.
- 🐛 **No-GNN MLP loader** now uses `torch.utils.data.DataLoader` for its `TensorDataset` (was PyG's DataLoader — worked on PyG 2.8, latent portability hazard).
- 🐛 **MLP smoke ternary fixed** (`50 if SMOKE else 1000`): smoke MLP epochs are now reduced like the GNNs' instead of inflated.
- ✨ **OLS slope (β) gate added**: OLS coefficient of normalized `sink_mean` persisted in `ols_floor` / `convergence_gates` / `s2_convergence_decision`, with gate |β−1| ≤ 0.1 (expected ≈ corr(analytical, MC) ≈ 0.97).
- ✨ **sW/sL cross-ratio added** to Tier A diagnostics: expected −L_nom/(2·W_nom) = −0.25; observed −0.2500 ± 0.0 exactly.
- ✨ **k-placement honesty**: identities now persist `k_value` and whether the test can discriminate placement at all (non-discriminative at k=1.0 — reported, not tuned).
- 🐛 **vdd single-sourcing**: `preflight()` and Tier A magnitude identities read `timing_params.vdd_v` (the source `delay_partials` consumes) instead of the unused `variation_params.vdd_v` default.
- ✨ **Training histories persisted** in the final results JSON (previously only in interim saves).
- 🐛 **Pelgrom effective-dimension cap unified**: `variation/analytical.py` now applies the same `min(d_um, 5.0)` cap as `variation/sampler.py`. Verified numerically inert on current parameters (max d ≈ 4.12 < 5).
- ➕ **`verify_stage6c.py` committed** to the repo root: recomputes all aggregates + sha256 digests from per-graph arrays without bulk transmission; accepts an artifact path argument (works on interim/smoke files).
- 🧪 Smoke run of the hardened runner passes end-to-end (exit 0); protected full-run artifacts and checkpoints verified byte-identical (md5) after the smoke run; 11/11 tests pass; Stage 3 locked reference still reproduces exactly.
- 🔁 **Full re-run on hardened code**: all 9 per-graph sha256 digests bit-identical to the pre-hardening run (hardening provably training-neutral); derived status reports `lockstep=exact`, S2 rel +39.0%, physics timing 0.68 ms/graph; new fields persisted (histories, OLS β = 0.9267 with gate PASS, sW/sL = −0.25, k-placement flag). Prior artifacts backed up to `gnn_baseline/{results,checkpoints}_backup_pre-hardening/` (gitignored).
- 📄 Session report relocated to `docs/postmortems/postmortem-stage6c-label-misalignment-and-verified-rerun.md`; audit + hardening documented in `docs/postmortems/postmortem-stage6c-runner-hardening.md`.

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
