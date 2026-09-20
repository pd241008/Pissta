# 📜 Changelog

> All notable changes to this project will be documented in this file.

---

### September 20, 2026 — propagate_path covariance defect fixed; Stage 4/4b/5 artifacts regenerated

- 🐛 **Fixed within-path covariance accumulation** (`ssta/statistical_sum.py`): `propagate_path` paired each new gate's covariance against `path[0]` only, dropping middle-gate cross terms on 3+-gate paths — on the six-gate §IV topology the missing term is 2·Cov(G2,G4) = 0.007990 (10.3% of path variance). The fixed accumulator pairs each new gate against the entire accumulated set (per-gate `var_d` on the diagonal, `cov_d` off-diagonal); bit-identical for 2-gate paths. `covariance_between_paths` (all-pairs) was never affected — cross-path covariance and cov_max_d6 were always correct.
- 🧪 **Regression tests added** (`tests/test_statistical_sum.py`): 2-gate backward compatibility vs the old accumulator, 3-gate closed-form all-pairs check with per-gate independent variance, frozen-config known-value pin. Full suite: 26 passed.
- 📊 **§IV numbers moved toward MC truth** (all comparisons vs locked N=100k seed-42): AT_G4 std bias −5.90% → **−1.19%**; final std −5.51% → **−3.88%**; final mean −0.54% → **−0.43%**; P99.87 −2.87% → **−2.59%** (analytical 10.4454 → **10.4756** vs MC 10.7536). Multi-seed P99.87 errors: −0.278 / −0.261 / −0.277 (seeds 42/123/999).
- 📐 **§X-B decomposition**: total gap 0.3083 → **0.2781** = shape 0.2083 (unchanged) + linearization **0.0698** (was 0.1000) — the bug was **30.2%** of the old linearization gap. Run B (empirical moments) untouched by construction (P99.87 10.5454 bit-identical). Disclosed side effect: ρ(AT4,AT5) 0.3618 → 0.3281, away from empirical 0.3846 — normalization artifact: the numerator Cov keeps its own separately-disclosed ~16% linearization deficit while the corrected denominator shrinks.
- 🔁 **Downstream artifacts regenerated in dependency order** (stage4 → stage4b → stage5) and re-verified: fresh-vs-stored 17/17 bit-exact; kit3 audit re-run **CLEAN** (D1/D2 absent, gap decomposition reproduced bit-exactly through the fixed code); stage5 closure gate still passes (54.3% / 57.3% of shape gap); stage3 locked MC reference untouched.
- 📝 **Docs updated**: `results/stage4_final_report.md` (§12 correction addendum), `results/stage5_final_report.md` tables, README Stage-11 row, kit3 audit narrative literals.

### September 19, 2026 — Cross-method follow-ups: OOD-100 accuracy comparison, pissta-10k scale test, Stage 4 audit (Kits 1–3)

- 📊 **OOD-100 cross-method comparison** (`results/stage9_ood_crossmethod_report.md`, `diagnostics/kit1_ood_export.py`): the point-accuracy counterpart to Stage 7's coverage collapse. On the same 100 OOD graphs (15–25 gates), GNN point error degrades ~6× (vanilla 2.5705 / maxbias_cm 2.3818 vs ≈0.37–0.43 ID) while the **validated analytical baseline (fixed_capped, 0.1129 ID) degrades only to 0.2111** — paired cluster bootstrap (Stage 6C methodology, 100 graphs × 3 seeds): GNN−analytical Δ +2.36 / +2.17, CIs [+1.96,+2.79] / [+1.81,+2.56], analytical wins 99–100% of graphs. GNN error grows monotonically with nrecon (≈5.0 at nrecon=5, no training analog) — same exchangeability-failure gradient as the coverage table. Per-graph predictions exported in the stage6c schema (`gnn_baseline/results/kit1_ood100_results.json`).
- 📈 **pissta-10k scale test** (`results/stage10_scale_test_report.md`, `diagnostics/kit2_train_10k.py` + `kit2_analyze_10k.py`): generation config confirmed identical to pissta-2k (seed 42, MC N=10k seed 42, gates 6–14, name-aligned sampler; 2k prefix byte-identical). Fresh 10k training (frozen protocol, 3 seeds): vanilla 0.2945, maxbias_cm 0.2891 — both significantly worse than the analytical baseline (0.1101, Δ +0.184/+0.179, CIs exclude 0); zero-shot 2k checkpoints: 0.4288/0.4197 (5× data helps the GNN significantly but leaves it ~2.6× behind analytical). **The Discussion's "may erode at scale" hypothesis is rejected** — the gap narrowed only from ≈0.26 to ≈0.18 while analytical inference stays at 0.498 ms/graph (p95 0.861, ≈linear in gate count; 1,505-graph test split in 0.75 s). Secondary finding: **the MAX-bias architectural advantage washes out at 10k** (Δ −0.0054, CI [−0.0155,+0.0048], ns, vs −0.062 sig. at 2k) — with a large corpus the simpler vanilla surrogate is the right choice.
- 🔎 **Stage 4 fixed-topology audit — CLEAN** (`results/stage11_stage4_audit_report.md`, `diagnostics/kit3_stage4_audit.py`): the same D1/D2 defect-detection method applied to the six-gate §IV pipeline finds both patterns absent (topo order == coords order and readback bit-invariant under all 720 gate_coords permutations; Pelgrom cap inactive, max gate distance 4.0 µm < 5 µm boundary). A from-scratch name-keyed reimplementation reproduces the stored analytical figures (P99.87 10.4454) and the full 0.3083 gap decomposition (0.2083 shape + 0.1000 linearization) **bit-exactly** from the locked N=100k seed-42 reference; 3 intermediates differ by ≤3.3e-16 (last-ulp numpy reduction-order artifact, not a modeling difference). Closes the open scope caveat: §IV rests on a verified pipeline.
- 📦 **Handoff packages** (`deliverables/`, produced by `diagnostics/package_deliverables.py`, sha256-manifested): `kit1_ood100/` (schema-verified ood_dataset.pkl + all results), `kit2_pissta10k/` (pissta-10k dataset.pkl/splits + fresh/zero-shot results), `kit3_stage4_audit/` (audited code + stored outputs + verdict).

### September 17, 2026 — Zenodo release kit: nested pissta-5k/10k companions, repo-root LICENSE, docs scope

- 📦 **Release kit in `zenodo/`** (ADR-010): `generate_pissta.py` + `verify_release.py`, per-version manifests with process constants, generator commit, config snapshot, real SHA256s, and toy-model/emulation disclaimers; `.zenodo.json` metadata prefill at the repo root. Four builds verified green: pissta-2k (frozen; all 2,000 MC labels recomputed bit-exactly), pissta-5k, pissta-10k, pissta-ood100.
- 🧬 **Nested companions**: the 2k build's gate-load↔gate-name pairing depended on the build process's `PYTHONHASHSEED` (set-iteration order — same nondeterminism class as B1/B4), so a fresh superset run would have silently changed the paper graphs' labels. 5k/10k therefore copy graphs 0–1999 verbatim from the frozen 2k and extend with a canonical (sorted) pairing; the N1 nesting check asserts byte-identity across 2k ⊂ 5k ⊂ 10k over all 2,000 prefix graphs.
- ⚖️ **License split**: MIT (code) + CC-BY-4.0 (data) in a single root-level `LICENSE`; `zenodo/LICENSE` is now a pointer stub.
- 🧹 **Docs scope**: `PROJECT_HISTORY.md`, `physics_informed_ssta_context.md`, and `docs/postmortems/` untracked (working notes, kept locally); `docs/adrs/` remains the canonical decision record — new **ADR-010** documents the release-kit decisions, and dangling references in reports/ADRs were redirected to ADRs and `results/stage8_report.md`.
- 🛡️ **Path hygiene**: release manifests and script output emit repo-relative paths only; an external `--out-dir` renders as `<external>/<name>` (no absolute-path leakage in published artifacts).

### September 13, 2026 — Stage 8 prep: training-time instrumentation fixed, combined MAX-bias + Tier A+B ablation, full cross-method cost table

- 🐛 **Training-time logging fixed** in the harnesses that silently dropped it (same bug class as B3): `run_maxbias.py`, `run_nocoor.py`, `run_tier_b_only.py` now persist `train_time` + `history` per seed and `avg_train_time_s` / `avg_inference_ms_per_graph` in their `stability_summary` (was: training finished, `train_result['train_time']` discarded at JSON write). `numpy` import added where the new aggregation needed it.
- ⏱️ **MAX-bias re-runs for instrumentation** (same hyperparameters/seeds as committed, deterministic CUDA): MAX-bias-CM 7 seeds and MAX-bias full-cap 3 seeds. **Per-graph lockstep vs the committed artifacts: max diff 0.0** — same experiment, now with real numbers. Training: vanilla avg 106–154 s, maxbias_cm avg 112 s, maxbias full-cap avg 120 s (single GPU, wall-clock, load-sensitive ±35% — bit-reproducible in metric, not in wall-time).
- 🔀 **Combined ablation executed** (`run_maxbias_tierab.py`, 7 seeds): MAX-bias-CM architecture (h=54, capacity-matched) **+** redesigned per-node Tier A+B features. Result: 0.3607 vs vanilla 0.4288, Δ **−0.068, CI [−0.105, −0.032]** (sig. better, 7/7 seeds negative) — but vs MAX-bias-CM **alone** Δ −0.006, CI [−0.033, +0.022] (**ns**, sign-unstable), at a 0.56 ms/graph physics-feature inference tax. **Combination not worth taking** — feature injection adds nothing even on top of the working architecture.
- 📊 **Stage 8 cross-method table** (`results/stage8_report.md`): MC / analytical (Clark) / tail-aware skew-normal / vanilla / MAX-bias (CM + full-cap) / combined × mean MAE / inference ms / training s / amortized data-generation cost. Uses real measured numbers only; tail-aware row explicitly marked n/a on the Stage-6 benchmark (only ever implemented for the fixed 6-gate DAG). Analytical baseline corrected to the current-dataset value **0.6609** (0.5911 in older docs was a stale earlier-dataset number, confirmed by direct recomputation).
- 🔬 **Independent verification**: point estimates recomputed from raw `per_graph` arrays (match to 6 decimals); CIs re-derived with fresh clustered-bootstrap code (RNG seed 7, 10 000 resamples) — all excluded-0 "vs vanilla" intervals confirmed, combined-vs-CM null confirmed; lockstep 0.0.
- ⚖️ **Honest amortized-cost framing** stated in the report: GNNs need ~210–260 s up front (102.5 s MC corpus + ~110 s training) before the first query; per-query is ~60–900× cheaper ONLY after that; the training premium roughly doubles the MC labeling bill. Breakeven ≈ 4,000 (N=10k) / ≈ 1,100 (N=100k) graphs. "Much lower cost than MC" is load-bearing in the amortized regime, not per lifecycle.

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
