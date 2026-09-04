# 📜 Changelog

> All notable changes to this project will be documented in this file.

---

### August 31, 2026 — Stage 7: split conformal calibration — the "calibrated uncertainty" half of the thesis

- 🎯 **Split conformal + studentized-residual score** (`gnn_baseline/conformal.py`, `run_stage7.py`): `score=|y−mean_pred|/std_pred`, `q_hat` from a held-out calibration set, interval `mean_pred ± q_hat·std_pred`, distribution-free (no Gaussian assumption). Carved the test split (304) into **cal 152 / eval 152**, stratified by nrecon (`carve_seed=7`), so coverage is asserted on an eval slice distinct from the Stage 6C full-test MAE points.
- 📊 **Pooled coverage @90% nominal (3 seeds):** **maxbias_cm 0.908** (meets target) · vanilla 0.895 (marginally under). Eval MAE: **maxbias_cm 0.371 vs vanilla 0.416** — the architectural win reconfirms on the fresh eval slice.
- ⚠️ **Disaggregated finding — both backbones under-cover nrecon=2** (vanilla 0.839, maxbias 0.865 vs 0.90): the reconvergence regime where MAX-inflated tails are the point. The pooled 90% hides this; reported per-bucket (see ADR-009 / `results/stage7_report.md`). **A better point estimate does NOT imply better calibration:** maxbias's tighter intervals (q_hat≈1.04 vs vanilla≈1.14) are what cause its nrecon=2 under-coverage.
- 🚫 **No coverage claim under distribution shift** — eval is same nrecon 1–4 family as cal; an OOD (unseen-topology-family) test is a documented follow-up, not claimed.
- 🧪 **MAX-bias CM checkpoints were not on disk** → both backbones retrained fresh identically (seeds [42,123,999]) for a fair calibration comparison. 19/19 tests pass (8 new conformal unit tests).

### August 29, 2026 — Stage 6C: redesigned features still no headroom; **architectural MAX-bias is the first positive result**

- 🔁 **Redesigned feature attempt** (ADR-008 addendum 1): Tier A now uses `var_d/load_ff²` (non-redundant vs `load_ff`; R²=0.088) and Tier B is per-node AT, passed through message passing. Still no headroom: Tier A (redesigned) **sig. worse** (+0.048, CI [+0.019,+0.078]); Tier A+B (redesigned) **ns** (−0.007, CI [−0.047,+0.032]); Tier B only **ns** (+0.009, CI [−0.027,+0.043]). Four feature-injection attempts in total — none beat vanilla.
- 🎯 **Architectural lever (the win): MAX-biased aggregation** (`aggr=["mean","max"]` at message passing, mirroring SSTA's own MAX-at-reconvergence). Full-capacity (h=64, +41% params): Δ −0.064 sig. better. **Capacity-matched (h=54, +1.1% params, 7 seeds): Δ −0.062, CI [−0.088,−0.037], 7/7 seeds negative** — the first defensible, capacity-controlled, multi-seed-stable positive result in the Stage 6C arc. Aggregate maxbias_cm 0.3666 vs vanilla 0.4288. The +41% params added ~nothing (matched Δ ≈ unmatched Δ), so the effect is architectural, not capacity. See ADR-008 addendum #2.
- 🧪 **Extended-seed + lockstep-by-seed fix**: replaced `compute_lockstep_verification`'s positional zip with seed-value matching; extended seeds without a 6B reference are recorded non-comparable (latent-only bug — every prior bit-exact run used exactly [42,123,999], sorted order == seed order).
- 🔬 **Vanilla-minus-coordinates diagnostic** (`run_nocoor.py`, 3 seeds): dropping x,y degrades vanilla ~3× (Δ +0.83, CI [+0.72,+0.95]) — vanilla does NOT reconstruct structure from `load_ff` alone; the headroom was in how fan-in/sibling info is combined (aggregation), not in injected features.
- 🧪 7-seed CM run: all 7 seeds negative; CI narrowed vs 3-seed (0.051 vs 0.057 width) rather than ballooning — signature of a real effect. 11/11 tests still pass.

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
