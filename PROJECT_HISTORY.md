# PROJECT HISTORY — Physics-Informed Surrogate SSTA (Pissta)

> **Complete chronological record of the project, from inception through Stage 6C.**
> **Compiled:** 2026-08-28, from git history, CHANGELOG, all ADRs, all postmortems, and the verified numeric artifacts.
> **Purpose:** a single self-contained history document covering every stage, decision, bug, fix, and result — so the full arc of the research is recoverable without reading scattered files.
>
> **Authoritative current state** (for status/next steps, see `physics_informed_ssta_context.md`). Key result: **feature-level physics injection (Tier A, Tier A+B) is significantly WORSE than the vanilla DAG-GNN baseline** on the verified artifact (CIs exclude 0).

---

## Table of Contents

1. [Project overview & research question](#1-project-overview--research-question)
2. [Research philosophy](#2-research-philosophy)
3. [Complete timeline](#3-complete-timeline)
4. [Stage-by-stage history](#4-stage-by-stage-history)
5. [Complete bug catalog (what went wrong)](#5-complete-bug-catalog-what-went-wrong)
6. [Architecture decision records (ADR summary)](#6-architecture-decision-records-adr-summary)
7. [Locked reference numbers](#7-locked-reference-numbers)
8. [Verification methodology (the pattern that caught the bugs)](#8-verification-methodology-the-pattern-that-caught-the-bugs)
9. [Known documentation inconsistencies](#9-known-documentation-inconsistencies)
10. [Next steps](#10-next-steps)

---


## 1. Project overview & research question

**Working title:**
Physics-Informed Surrogate SSTA for Correlated Process Variations with Calibrated Worst-Case Timing

**Core research question:**
> Can a physics-informed surrogate approximate Monte-Carlo-quality worst-case timing under correlated process variations at much lower computational cost, while also providing trustworthy uncertainty estimates?

**Central claim the paper is building toward:**
> Physics + statistical timing + ML surrogate + calibrated uncertainty can approximate Monte Carlo-level worst-case timing at much lower computational cost.

**Scope & honesty constraints (govern every stage):**
- Clearly separate demonstrated results from planned results.
- The paper must NOT claim (until actually demonstrated): exact Gumbel corrections, fully non-Gaussian propagation, guaranteed coverage under arbitrary distribution shift, or any performance numbers not experimentally reproduced.
- Compare everything against Monte Carlo; run ablations; test unseen topologies; report tail error (P95/P99/P99.87); report runtime and data-generation cost separately; use physics constraints as an actual mechanism.

**Toy-model disclaimer:** all numerical values (α=1.3, Vdd=1.0V, L_nom=45nm, W_nom=90nm, Vth_nom=0.40V, etc.) are deliberately simple demonstration parameters for software/methodology validation — **not silicon-calibrated**, and never to be presented as real technology results.

---

## 2. Research philosophy

Do not: fabricate novelty/results/acceptance; call an approximation "exact" without stated assumptions; label a model non-Gaussian if the implementation is Gaussian; hide ML/data-generation costs; add a technique without an experiment justifying it.

Do: compare everything against MC; ablate; test generalization to unseen circuits; report tail + runtime + cost separately; use physics constraints as a real mechanism; validate every stage before building the next.

**Verification discipline (proven essential): every reported result gets independently re-derived from raw files/arithmetic before being trusted** — not just read and accepted. This discipline, applied consistently, caught every bug in Section 5.

---

## 3. Complete timeline

| Date (2026) | Event |
|---|---|
| **Aug 14** | Stage 1 baseline: 5-gate feed-forward chain MC. Headline: mean=13.6251, std=0.8824, P99.87=16.6037. |
| **Aug 15** | Stage 2 (correlated variation), Stage 3 (branching DAG locked MC reference, N=100k, seed 42). Beginnings of Stage 4. Postmortems: critical-path-hardcoding, numpy-erf. |
| **Aug 15** | Stage 4 analytical SSTA (Clark MAX + linearized delay). P99.87=10.4454, 566× speedup. |
| **Aug 16** | Stage 4b gap decomposition (shape=0.2083, linearization=0.1000). Stage 5 tail-aware skew-normal MAX. Monorepo restructure per Design Dungeons. Postmortems: stage5-variable-shadowing. ADRs 001–005. |
| **Aug 17** | Stage 6A dataset generation, 4 rounds of bug-fixing (Vth singularity, graph loss, stale stats, thin coverage, sink-predecessor, timing accounting). ADR-006. Postmortem: stage6a-critical-issues. |
| **Aug 19–20** | Stage 6B vanilla DAG-GNN baseline. ADR-007. Postmortem: stage6b-critical-issues (7 issues). 27 consolidated review issues addressed. |
| **Aug 20** | Stage 6C physics-informed DAG-GNN 3-way ablation, **Round 1** (null + significant Tier A+B −21% claim). ADR-008. |
| **Aug 20–21** | Bug-Fix Sweep: **B1 MC label misalignment** (+B2–B6). Deterministic CUDA re-run. **Round 2** result reverses → sig. worse. Session report → `postmortem-stage6c-label-misalignment-and-verified-rerun.md`. |
| **Aug 21** | Runner Hardening audit (N2/N4/N5/N6, Pelgrom cap), `verify_stage6c.py`, OLS β gate. Full hardened re-run. Postmortem: stage6c-runner-hardening. Interim null result documented in some docs. |
| **Aug 24** | Commit `9153685` "feat(stage 6c) finished with null results to be redone and investigated". Postmortem: stage6c-honest-assessment (null framing). |
| **Aug 28** | Context/ADR desync discovered and resolved (user decision: **sig. worse is authoritative**). New context file created. This history document compiled. |

> **Timeline note:** the git-tracked docs (Aug 21–24 postmortems) record a *null* Stage 6C result (vanilla 0.4263, CIs crossing 0), while the committed numeric artifact `gnn_baseline/results/stage6c_results.json`, the CHANGELOG, and the README record a *significantly-worse* result (vanilla 0.4082). Per explicit user decision the artifact-backed **sig.-worse** reading is authoritative. See §9.

---

## 4. Stage-by-stage history

### Stage 1–2 — Chain-only baseline + correlated variation
- Simple gate-chain SSTA with correlated variation (inter-die + spatial PCA + Pelgrom random).
- Established that spatial correlation **widens the tail** (P99.87: 14.89 → 15.43 in the toy chain) with little effect on the mean.
- Stage 1 baseline: mean=13.6251, std=0.8824, P95=15.1103, P99=15.7961, P99.87=16.6037 (N with 5-gate chain).
- Files: `physics_informed_ssta_stage1.zip`, `physics_informed_ssta_stage2.zip`.

### Stage 3 — Monte Carlo reference for a branching DAG (locked)
- Built a 6-gate branching/reconvergent DAG:
  ```
                ┌── G2 ── G4 ──┐
  Input → G1 ──┤                ├── G6 → Output
                └── G3 ── G5 ──┘
  ```
- **N=100,000, seed 42 locked as ground truth.** Reference: mean=9.3065, std=0.4194, P95=10.0243, P99=10.3675, **P99.87=10.7536**.
- Reproducibility across seeds 42/123/999: Δ(branching−chain) P99.87 = 0.3255 ± 0.0120.
- Confirmed MAX-induced tail inflation is real (branching P99.87 > naive Gaussian mean+3σ 10.5648 by ~1.8%).
- Asymmetric sanity check passed (100% critical-path selection → validated AT/argmax logic). Critical-path split ~49.9%/50.1%.
- Raw data persisted (`stage3_raw.npz`). ADR-001 explains N=100k choice (tail convergence; N=10k too noisy).

### Stage 4 — Analytical SSTA (Clark MAX + linearized delay)
- Closed-form pipeline: linearized alpha-power delay → Gaussian SUM → Clark MAX at reconvergence → SUM with G6.
- **P99.87 = 10.4454** vs MC 10.7536 → **gap 0.3083 (2.87%)**, consistent across seeds.
- **Speedup 566× vs 100k MC** (0.35ms vs 200ms).
- Covariance bookkeeping verified against empirical MC (within 3–5%).
- ADR-002 (Clark MAX choice), ADR-003 (first-order linearization choice) document the choices and their known limitations (skew + linearization under-estimate the tail).

### Stage 4b — Gap decomposition (hybrid ablation)
- Fed Clark's MAX the **empirical** (not linearized) path moments, isolating error source:
  ```
  total_gap (0.3083) = shape_gap (0.2083) + linearization_gap (0.1000)
  ```
- **~68% shape error, ~32% linearization error.** → quantified justification for Stage 5 (attacks the bigger piece).

### Stage 5 — Tail-aware SSTA (skew-normal 3-moment MAX)
- Replaced Clark's 2-moment Gaussian MAX with a 3-moment skew-normal fit (empirical skewness of max(AT_G4,AT_G5) = 0.2475).
- **Tail-aware P99.87 = 10.6584**; absolute gap mean 0.089 ± 0.009 across 3 seeds; pooled-MC closure ≈ 57%. Runtime ~140ms (~1.4× vs MC).
- **Methodological fix:** replaced seed-dependent closure% (swung 54–62% due to MC reference noise) with absolute-gap primary + pooled-MC closure secondary. ADR-005.
- Bug fixed: `mc_p99_87` variable shadowing in multi-seed loop (see §5).

### Stage 6A — GNN training data generation (full pipeline)
- Dataset of (random DAG topology → MC-labeled delay mean/std) for GNN training on **arbitrary** graphs.
- `graph_generator.py` (structured random generation with validation), generalized `monte_carlo.py`, `analytical_ssta_arbitrary.py` (iterative Clark MAX).
- **Final locked dataset:** 2000/2000 valid graphs, 0 discards. Split 1397/296/307 (original; later regenerated to 1398/298/304, nrecon 1–4), stratified by reconvergence. Gate sizes 6–14. Label noise mean ≈0.05%, std ≈0.7–0.8%. Mean delay 8–29, std delay 0.38–1.16.
- Physics features stored per graph: per-gate linearized sensitivities ∂d/∂{L,W,Vth} + analytical-SSTA baseline.
- Went through **4 rounds of bug-fixing** (see §5): Vth/Vdd singularity, undocumented graph loss, stale summary stats, thin complex-topology coverage, sink-predecessor mismatch, timing accounting.
- ADR-006; postmortem-stage6a-critical-issues.

### Stage 6B — Vanilla DAG-GNN baseline
- Raw-features-only (load_ff, x, y — no physics) GraphSAGE baseline to predict per-graph critical-path (mean, std).
- **Architecture:** 3× GraphSAGE (SAGEConv→BN→ReLU→Dropout, hidden=64) + mean-pool readout + 2-layer MLP head → 2 outputs; 29,698 params; Adam lr=1e-3; MSE; early stopping (patience=20); grad clip 1.0; directed edges. ADR-007.
- **Data hygiene:** normalization from train split only; frozen `splits.json`; no re-shuffling.
- **Results (original set):** mean MAE 0.6873±0.0042 (4.04%), std MAE 0.03375±0.0002 (4.67%). Beats analytical SSTA on mean (ratio 0.95) and substantially on std (ratio 0.37) — 3/3 seeds. Beats trivial baseline by large margin. Inference 0.08 ms/graph.
- **7 critical issues found & fixed** (see §5): checkpoint provenance, 32× timing error, loss misdescription, confounded overfitting check, missing batch check, nrecon mean contradiction, fragile dual-construction metadata.
- Postmortem-stage6b-critical-issues; ADR-007.

### Stage 6C — Physics-informed DAG-GNN (3-way ablation) — THE HEADLINE
3-way ablation with frozen architecture/hyperparameters/splits/seeds; only input features vary.
- **Vanilla** = 6B locked. **Tier A** = node features 3→6 dims (+[∂d/∂Vth, ∂d/∂L, ∂d/∂W]). **Tier A+B** = Tier A node + 2 graph-level scalars (analytical sink_mean, sink_std) appended after pooling.
- Mandatory diagnostics: Tier A redundancy check, Tier B leakage check, no-GNN residual-MLP baseline, lockstep verification, paired cluster-bootstrap CIs.

**Reversal saga across three rounds (a documented lesson):**
1. **Round 1 (original, invalidated):** Tier A null; Tier A+B −21% mean-MAE, reported significant (CI excluded 0). Later found to be an artifact of mislabeled data.
2. **Round 2 (critical bug B1 found):** labels were wrong → corrected result **reversed**: both tiers significantly *worse* than vanilla (Tier A +0.040, Tier A+B +0.063 vs vanilla 0.408).
3. **Round 3 (hardening audit + full re-run):** independently audited; the committed artifact yields **significantly WORSE for both tiers** (CIs exclude 0). Note: some git-tracked docs describe an "interim" null (see §9).

**Verified final numbers (3 seeds, N=304 test graphs), from committed `stage6c_results.json`:**
| Model | Mean MAE | vs Vanilla (CI) | Mean rel. | Significant? |
|---|---|---|---|---|
| Analytical SSTA | 0.5911 | — | — | — |
| **Vanilla DAG-GNN** | **0.4082 ± 0.0055** | — | 2.46% | — |
| Tier A | 0.4478 ± 0.0220 | +0.040 [+0.0241, +0.0557] | 2.69% | **Yes — worse** |
| Tier A+B | 0.4714 ± 0.0095 | +0.063 [+0.0147, +0.1123] | 2.97% | **Yes — worse** |
| No-GNN Residual MLP | 0.6553 ± 0.0031 | — | 4.21% | — |

**Diagnostics:**
- Lockstep (6B → 6C vanilla): **bit-exact** (max diff 0.0, all seeds) post-B4.
- Tier A structurally redundant: ∂Vth/∂L = 97.5, ∂W/∂L = −0.25 (algebraically fixed ratios of `load_ff`), carrying **zero information beyond load_ff** for any graph.
- Tier B leakage: corr(sink_mean, label) ≈ 97%, OLS β = 0.9267.
- No-GNN MLP (0.655) ≫ all GNN variants → graph structure matters, but explicit physics features don't help (they hurt).

---

## 5. Complete bug catalog (what went wrong)

### Stage 3-era
- **[postmortem-critical-path-hardcoding]** Critical-path tracker hardcoded G4/G5 as merge predecessors → async sanity check gave 50/50 instead of 0/100. Fixed with dynamic sink/pred detection + `np.argmax`.
- **[postmortem-numpy-erf]** `np.erf()` AttributeError on NumPy 2.x. Fixed with `math.erf()`.

### Stage 5
- **[postmortem-stage5-variable-shadowing]** Loop var `mc_p99_87` shadowed the locked seed-42 reference; JSON field silently showed seed 999's number. Fixed by renaming to `mc_p99_87_seed`.

### Stage 6A (4 rounds)
1. **Vth/Vdd singularity** — unclipped Vth let `(Vdd−Vth)→0`, catastrophic delay blowups (mean_delay up to 25,313). Fixed via Pelgrom distance cap + physical Vth bounds + raised alpha-power floor (1e-6 → 0.1V).
2. **Undocumented graph loss** — 652/2000 graphs silently discarded (chain-only, no reconvergence). Fixed: manifest reports `n_generated`/`n_dataset`/`skip_reasons`.
3. **Stale summary_stats.json** — computed on pre-filter set. Fixed to compute from actual saved dataset.
4. **Thin complex-topology test coverage** — only 3 nrecon≥3 in test. Fixed with wider n_gates + min_reconvergence at generation.
5. **Sink-predecessor mismatch** — generator's check didn't match pipeline's sink requirement. Fixed by forcing first build op to be split-reconverge on source→sink (0 discards post-fix).
6. **Timing-accounting artifact** — `total_generation_time_s` included construction overhead (total/mean ≈ 2020 ≠ 2000). Fixed by timing only the per-graph loop.

### Stage 6B (7 issues)
1. Cement checkpoint/model provenance mismatch (`.module.` BN keys) — resolved/round-trip verified.
2. Inference timing off by ~32× (batch-level reported as per-graph, no warm-up, `time.time()`).
3. Loss misdescribed in report (sum vs mean reduction).
4. Overfitting check confounded by train/eval mode mismatch — added `eval_mode_train_loss`.
5. Missing PyG batch shape-check (spec-mandated, never implemented).
6. nrecon mean contradiction (generator param 2.85 vs realized 3.31).
7. Fragile dual-construction metadata pattern — switched to single source of truth from `loader.dataset`.

### Stage 6C (the sagas)
- **B1 (critical):** `ssta/monte_carlo.py` indexed variation sample columns by topological position but populated in `gate_coords` key order → arbitrary DAGs received other gates' variation samples → corrupted MC labels across the whole dataset. Fixed by name-based `col_idx` mapping. No-op for Stage 3's locked ref (exact match post-fix), which is why earlier stages were never affected.
- **B2:** latent `NameError` (`build_branching_graph` never imported) in default-graph MC path.
- **B3:** Tier A+B physics timing silently zeroed by `except KeyError: pass` (gate name mismatch) — never measured the analytical SSTA.
- **B4:** CUDA non-determinism — `cudnn.deterministic=True` insufficient; scatter-add atomics non-deterministic regardless. Fixed with `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Root-caused via multi-step investigation (see §8).
- **B5:** CWD-dependent default config/dataset paths.
- **B6:** dead `preflight()` never called.
- **N2:** smoke runs could overwrite real artifacts → smoke gated + separate artifacts.
- **N4:** PyG DataLoader over plain TensorDataset (MLP) → portability hazard.
- **N5:** inverted smoke-mode epoch budget (MLP got 20×).
- **N6:** hand-edited "verified" status string could desync → derived at runtime.
- **Pelgrom cap:** `sampler.py` capped d at 5.0 but `analytical.py` didn't → unified (numerically inert at current params).
- **[Step 1 correction, 2026-08-29]** First redundancy probe for redesigned Tier A used the *current* process-moment formula (variation/analytical.py, changed 08-22) instead of the dataset's stored delay_var (built 08-21) — showed spurious corr=0.88 vs load_ff. The dataset's own delay_var was what training actually consumed; recomputed from that source gave the correct R²=0.088. Same class of bug as B1 (label/feature source mismatch across a pipeline version boundary) — caught before it reached training.
- **[compute_lockstep_verification positional-zip bug, 2026-08-29]** Lockstep check compared vanilla seeds by list position (`zip`) rather than by seed value; harmless while the seed lists matched 1:1 in order, but latent under the 7-seed extended run (extra seeds not present in the 6B reference — e.g. if 777 sorts between 123 and 999, 6B's 999 would be compared against 6C's 777). Fixed to match-by-seed explicitly, with extended seeds correctly excluded from the 6B comparison. **Confirmed non-retroactive:** every prior "bit-exact" claim (6B baseline, Tier A/A+B, Tier B-only, full-capacity maxbias, nocoor) compared exactly 3 vanilla seeds `[42,123,999]`, whose sorted order is self-identical, so positional `zip` ≡ seed-match for those runs (verified against the committed artifacts: all `all_match=True, max_mean_diff=0.0`). The bug could only manifest with ≥4 seeds whose sorted order reorders.

---

## 6. Architecture decision records (ADR summary)

| ADR | Topic | Decision |
|---|---|---|
| 001 | Stage 3 MC reference | N=100k, seed 42 as locked ground truth (tail-converged). |
| 002 | MAX approximation | Clark's Gaussian moment-matched MAX (correlation-preserving, O(1)). |
| 003 | Delay model | First-order Taylor linearization of alpha-power delay (standard SSTA approach). |
| 004 | Repo structure | Numbered monorepo: foundations→variation→timing→ssta→experiments. |
| 005 | Stage 5 reporting | Pooled MC reference + absolute gaps primary; closure% secondary. |
| 006 | Stage 6A data generation | Structured random generation with validation; sink-reconvergence by construction; physical bounds. |
| 007 | Stage 6B architecture | GraphSAGE (3 layers, hidden 64, mean pool, directed edges) over GCN/GAT. |
| 008 | Stage 6C feature injection | 3-way ablation feature-level physics injection (Tier A: node sensitivities; Tier A+B: + graph-level analytical). *Updated 2026-08-28 to reflect sig.-worse outcome.* |

---

## 7. Locked reference numbers

| Method | P99.87 | vs MC gap | Runtime | Notes |
|---|---|---|---|---|
| Monte Carlo (N=100k, seed 42) | **10.7536** | — | 200ms | Ground truth, Stage 3 |
| Naive Gaussian (MC mean+3σ) | 10.5648 | 1.88% | — | Stage 3 |
| Clark / linearized moments (Stage 4) | 10.4454 | 2.87% | 0.35ms (566×) | |
| Clark / empirical moments (Stage 4b) | 10.5454 | 1.94% | — | isolates shape_gap |
| Tail-aware skew-normal (Stage 5) | 10.6584 | 0.83% (pooled ~57% closure) | 137ms (~1.4×) | |

Stage 6 uses a **different** (arbitrary-topology, regenerated) dataset — GNN numbers are evaluated against per-graph MC labels (N=10k), not the single locked 6-gate reference, so they are not directly comparable to the table above.

**Stage 6C verified headline:** Vanilla 0.4082 · Tier A 0.4478 (+0.040, sig. worse) · Tier A+B 0.4714 (+0.063, sig. worse) · Analytical 0.5911 · No-GNN MLP 0.6553.

**2026-08-29 addendum:** ADR-007's std MAE figure of 0.0337 (4.67%) describes the pre-regeneration "original" 1397/296/307 split, not the current locked dataset. The current lockstep-verified pipeline (6B and all 6C variants, bit-exact) reports std MAE 0.0188 (2.56%). No drift — this is a stale-doc citation, corrected here. The 0.0337 figure should be removed or clearly scoped in ADR-007's prose table.

**2026-08-29 addendum #2 — architectural result:** MAX-biased aggregation, capacity-matched (h=54, 30,026 params vs vanilla's 29,698), 7 seeds: mean MAE 0.3666 vs vanilla 0.4288 (see per-seed table in ADR-008 addendum #2), Δ = −0.062 [−0.088, −0.037], sig. better, sign-stable 7/7. **First positive Stage 6C result.** The +41%-params full-capacity variant gave the same Δ (−0.064), so the extra capacity added ~nothing; the effect is ~−0.06 at matched or unmatched capacity. Vanilla-minus-coordinates control: Δ = +0.83 [+0.72, +0.95] confirms vanilla relies heavily on geometry, motivating why aggregation-structure change (not feature injection) was the effective lever.


---

## 8. Verification methodology (the pattern that caught the bugs)

1. Never trust a summary number — recompute from the actual raw file/array.
2. Cross-check every derived quantity against an independent formula path (finite-difference, etc.).
3. Check multi-seed stability before trusting a single result.
4. If an "improvement" depends on a noisy denominator (closure% vs single MC seed), report the absolute-difference version as primary.
5. When code changes are reported as fixes, re-verify against the *actual regenerated files*, not the stated diff.
6. **A headline result can reverse after a downstream bug fix.** Stage 6C went 21%-better → sig.-worse → (interim null in some docs) → sig.-worse. Check whether a CI includes zero before accepting a table's headline.
7. Ablation tables can misrepresent significance if read without their CIs — the "S2" (GNN vs no-GNN) and "vs vanilla" (physics vs vanilla) comparisons measure different things.
8. **Documentation can desync from code** — always cross-check doc claims against the committed artifact via `verify_stage6c.py`.

**The B4 root-cause investigation (exemplary forensic chain):**
train_loss bit-reproducible back-to-back on CPU → 6B-vs-6C val_loss diverged at epoch 0 → CPU replication gave a third trajectory → thread/allocator probes unchanged → re-running the script failed to reproduce itself → a debug print revealed `Device: cuda` (all probes had assumed CPU) → CUDA + deterministic-algorithm flags reproduced ep0 exactly and became bit-stable.

---

## 9. Known documentation inconsistencies

The repository's own documentation conflicts on the Stage 6C verdict:
- **Authoritative (artifact-backed):** `gnn_baseline/results/stage6c_results.json` (verified by `verify_stage6c.py`), `CHANGELOG.md`, `README.md`, `results/stage6c_report.md` banner, and the Context section of `postmortem-stage6c-runner-hardening.md` → **sig. worse** (vanilla 0.4082; Tier A +0.040; Tier A+B +0.063; MLP 0.6553).
- **Stale (null-framed):** `postmortem-stage6c-honest-assessment.md` and `postmortem-stage6c-label-misalignment-and-verified-rerun.md`, plus the Results section (§4/§5) of `postmortem-stage6c-runner-hardening.md` (internally self-contradictory vs. its own Context) → **null** (vanilla 0.4263; Tier A +0.0077; Tier A+B +0.0044; MLP 0.5786).

**Resolution (user decision, 2026-08-28):** the artifact-backed **sig.-worse** reading is authoritative. The null-framing documents are flagged as stale and should be updated (not yet edited). The earlier "significant −21% improvement" claim is universally acknowledged (across all sources) as an artifact of the B1 mislabeled dataset.

---

## 10. Next steps

**Current place in plan:** Stage 6C resolved (2026-08-29): the four feature-injection variants (original Tier A/A+B sig. worse; redesigned Tier A sig. worse; redesigned Tier A+B ns wide; Tier B only ns wide) added no headroom — but the **architectural pivot delivered: MAX-biased aggregation beats vanilla, capacity-matched, 7 seeds, CI entirely below 0** (the only positive result in the arc; see ADR-008 addendum #2).

**Planned next attempts, in priority order:**
1. ~~Redesign Tier A~~ — DONE (2026-08-29), still sig. worse (non-redundant ≠ useful).
2. ~~Redesign Tier B as per-node~~ — DONE (2026-08-29), ns/tied but CI wide (3-seed); tier_b_only isolation also ns with sign-unstable per-seed deltas.
3. ~~Architectural physics constraint~~ — DONE (2026-08-29): MAX-biased aggregation (mean+max fanin at message passing) beats vanilla, capacity-matched (h=54), 7 seeds, Δ = −0.062 [−0.088, −0.037]. **First positive result in the Stage 6C arc, and capacity-independent** (same Δ ≈ −0.06 at +41% params).
4. ~~Vanilla-minus-coordinates diagnostic~~ — DONE (2026-08-29): vanilla degrades ~3× without x,y (Δ = +0.83 [+0.72, +0.95]), confirming it does NOT implicitly reconstruct physics from load_ff alone — ties the architectural win back to the central research question (headroom was in aggregation of geometric/structural info, not in additional physics scalars).

**New next steps:**
5. Extend the honesty-constraint framing for the paper: the headline claim is now "architectural physics-consistent aggregation improves over vanilla; feature-level injection alone does not" — both halves need to be reported together, not just the win, per research philosophy (§2).
6. Consider whether MAX-bias + Tier A+B (redesigned, per-node, the closest-to-tied feature variant) combined has any headroom beyond MAX-bias alone — cheap follow-up now that a working positive architecture exists as the base to ablate features on top of.
7. Stage 7 (conformal calibration) can now proceed against the MAX-bias capacity-matched model as the strongest verified backbone, once the docs above are committed.

**Optional (tighter-CI honesty claim, now superseded as a priority):** extend the tied feature variants (Tier A+B, B-only) to 5–7 seeds to convert their "ns, wide" into "ns, tight". Given sign-unstable per-seed deltas already observed, this is expected to confirm the null rather than reverse it — still a materially stronger claim for the paper's honesty-constraint section, but no longer gating since a positive architectural result now exists.

**After Stage 6C is resolved:** Stage 7 (conformal calibration), Stage 8 (final cross-method comparison table with honest amortized data-generation cost accounting, and an explicit methods-section footnote documenting the Stage 6C bug-fix saga).

**Outstanding hygiene tasks:** update the two stale null-framing postmortems (and the internally-contradictory §4/§5 of `postmortem-stage6c-runner-hardening.md`); fix the `run_stage6c.py` "both tiers not-significant" console branch that still prints hardcoded delta text.
