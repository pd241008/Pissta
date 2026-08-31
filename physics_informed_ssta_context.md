# CONTEXT FILE — Physics-Informed Surrogate SSTA Research Project

**Status as of:** Stage 6C complete. Feature-level physics injection added **no headroom across four attempts** (original Tier A/A+B sig. worse; redesigned Tier A sig. worse, redesigned Tier A+B and Tier B-only ns) on a fully bug-fixed, independently-verified pipeline. The eventual win came from the **architectural** route: **MAX-biased aggregation beats vanilla, capacity-matched (Δ −0.062, CI [−0.088,−0.037], 7/7 seeds negative)** — the arc's first positive result (ADR-008 addendum #2). Supersedes the prior corpus which described the hardened result as feature-injection sig.-worse and recommended further feature redesign before architecture.
**Supersedes:** the prior version of this file (ended at "Stage 6C complete and hardened — null result"), which itself superseded the original `physics_informed_ssta_context.md` (ended at "Stage 3 — next task").

> **Verification note (2026-08-28):** All Stage 6C numbers in this file that differ from the prior file were re-derived from the committed numeric artifact — `gnn_baseline/results/stage6c_results.json` via `verify_stage6c.py` (aggregates recomputed from per-graph arrays; per-graph sha256 digests consistent; lockstep max diff 0.0) — and agree with `CHANGELOG.md`, `README.md`, `results/stage6c_report.md`, and the Context section of `docs/postmortems/postmortem-stage6c-runner-hardening.md`. `docs/adrs/ADR-008` was updated to match. **Documented-conflict resolution (2026-08-28, decision logged):** the repo's own docs were found to conflict on the Stage 6C verdict — two postmortems (`postmortem-stage6c-label-misalignment-and-verified-rerun.md` and the Results section of `postmortem-stage6c-runner-hardening.md`, which is internally self-contradictory vs. its own Context) still record a **null** result (vanilla 0.4263, CIs crossing zero) that does **not** match the committed artifact. Per explicit user decision, the **significantly-worse-to-vanilla** reading backed by the raw artifact (and README/CHANGELOG/report banners) is treated as authoritative; the two null-framing postmortems are **stale and should be updated** (flagged, not yet edited — a future task). The old null table is preserved below as the superseded interim re-run.

---

## 1. Project purpose

Research project in VLSI statistical timing analysis (SSTA).

**Working title:**
Physics-Informed Surrogate SSTA for Correlated Process Variations with Calibrated Worst-Case Timing

**Core research question:**
> Can a physics-informed surrogate approximate Monte-Carlo-quality worst-case timing under correlated process variations at much lower computational cost, while also providing trustworthy uncertainty estimates?

This is run as a real research workflow, not a paper-writing exercise. The assistant acts as a PhD-level VLSI research collaborator and is expected to be brutally honest about technical weaknesses, novelty, and claims — including finding and flagging bugs in the user's own code/data before results are trusted.

**Central claim the paper is building toward:**
> Physics + statistical timing + ML surrogate + calibrated uncertainty can approximate Monte Carlo-level worst-case timing at much lower computational cost.

**The paper should NOT claim** (until actually demonstrated): exact Gumbel corrections, fully non-Gaussian propagation, guaranteed coverage under arbitrary distribution shift, or any performance numbers not experimentally reproduced.

---

## 2. Research philosophy (governs every stage)

Do not:
- fabricate novelty, fabricate experiment results, claim publication/acceptance
- call an approximation "exact" unless mathematically exact under clearly stated assumptions
- call a model non-Gaussian if the implementation is Gaussian
- hide ML training/data-generation costs when reporting speedups
- add a fancy technique unless an experiment shows why it's needed

Do:
- clearly separate demonstrated results from planned results
- compare everything against Monte Carlo
- perform ablations
- test unseen circuits/topologies (generalization, not memorization)
- report tail error (P95/P99/P99.87), not just mean error
- report runtime and training/data-generation cost separately
- use physics constraints as an actual mechanism, not decorative language
- validate every stage (multi-seed stability, sanity checks) before building the next stage on top of it

**Verification discipline established over the project:** every reported result gets independently re-derived from raw files/arithmetic before being trusted — not just read and accepted. This has caught multiple real bugs (see Section 7), including a context-file/artifact desync (this revision).

---

## 3. Toy-model disclaimer

All numerical values (α=1.3, Vdd=1.0V, L_nom=45nm, W_nom=90nm, Vth_nom=0.40V, etc.) are deliberately simple demonstration parameters for software/methodology validation. They are **not silicon-calibrated** and must never be presented as real technology results.

---

## 4. Core formulas established (unchanged since original context file)

**Variation decomposition:**
```
X_i = X_nom + ΔX_inter + ΔX_spatial,i + ΔX_random,i
```

**Spatial covariance (exponential kernel):**
```
Σ_ij = σ² exp(-d_ij / λ)
```

**PCA sampling of spatial term:**
```
Σ = U Λ U^T
ΔX_spatial = U_r Λ_r^(1/2) z,   z ~ N(0,I)
```

**Pelgrom mismatch (Vth):**
```
σ²_Vth = A²_Vth/(W·L) + S²_Vth·D²
```

**Alpha-power gate delay (with geometry sensitivity):**
```
d = (C_load · Vdd) / [k · (Vdd − Vth)^α] · (L/L_nom) / √(W/W_nom)
```

**Statistical SUM (exact for Gaussians):**
```
μ_sum = μ_X + μ_Y
σ²_sum = σ²_X + σ²_Y + 2ρσ_Xσ_Y
```

**Statistical MAX (Clark's Gaussian moment-matching approximation):**
```
a = √(σ1² + σ2² − 2ρσ1σ2)
θ = (μ1 − μ2) / a
μ_max = μ1·Φ(θ) + μ2·Φ(−θ) + a·φ(θ)
σ²_max = (μ1²+σ1²)Φ(θ) + (μ2²+σ2²)Φ(−θ) + (μ1+μ2)·a·φ(θ) − μ_max²
```

**Quantile target:** P99.87 ≈ mean + 3σ under Gaussian assumption (used as the standard tail metric throughout).

---

## 5. Stage-by-stage summary

### Stage 1–2 (pre-DAG, chain-only)
Simple gate-chain SSTA with correlated variation (inter-die + spatial PCA + Pelgrom random). Established that spatial correlation widens the tail (P99.87: 14.89 → 15.43 in the toy chain) without much affecting the mean. Files: `physics_informed_ssta_stage1.zip`, `physics_informed_ssta_stage2.zip`.

### Stage 3 — Monte Carlo reference for a branching DAG ✅ Complete, locked
Built a 6-gate branching/reconvergent DAG:
```
              ┌── G2 ── G4 ──┐
Input → G1 ──┤                ├── G6 → Output
              └── G3 ── G5 ──┘
```
- N=100,000 Monte Carlo samples, seed 42 **locked as reference ground truth**.
- **Reference stats (seed 42):** mean=9.3065, std=0.4194, P95=10.0243, P99=10.3675, **P99.87=10.7536**.
- Reproducibility validated across seeds 42/123/999: Δ(branching P99.87 − chain P99.87) = 0.3255 ± 0.0120 (stable, ~27:1 signal/noise).
- Chain-vs-branching comparison confirms MAX-induced tail inflation is real (not an artifact): branching P99.87 (10.7536) > naive Gaussian mean+3σ (10.5648) by ~1.8% relative — first empirical evidence the tail is non-Gaussian.
- Asymmetric-DAG sanity check passed (100% critical-path selection on the structurally longer path — confirms AT/argmax propagation logic is correct).
- Critical-path split in the balanced DAG: ~49.9%/50.1% — genuinely exercises the MAX operation.
- Raw data persisted: `stage3_raw.npz` (100k arrival-time samples, per-gate L/W/Vth draws).

### Stage 4 — Analytical SSTA (Clark's MAX, linearized delay) ✅ Complete
Closed-form pipeline: linearized alpha-power delay moments → Gaussian SUM along paths → Clark MAX at reconvergence → Gaussian SUM with G6.
- **Analytical P99.87 = 10.4454** vs MC 10.7536 → **gap = 0.3083 (2.87% relative)**, consistent across all 3 MC seeds (0.291–0.308).
- **Speedup: 566× vs 100k-sample MC** (0.35ms vs 200ms).
- Covariance bookkeeping independently verified against empirical MC covariances (within 3–5%).

### Stage 4b — Gap decomposition (hybrid ablation) ✅ Complete
Isolated *why* Clark underperforms by feeding Clark's MAX formula the **empirical** (not linearized) path moments:
```
total_gap (0.3083) = shape_gap (0.2083) + linearization_gap (0.1000)
```
**~68% of the error is Gaussian-MAX-shape error, ~32% is linearization error.** This decomposition is the quantified justification for Stage 5 (attacks the bigger piece first).

### Stage 5 — Tail-aware SSTA (skew-normal 3-moment MAX) ✅ Complete
Replaced Clark's 2-moment Gaussian MAX with a 3-moment skew-normal fit (matches empirical skewness of max(AT_G4,AT_G5) = 0.2475).
- **Tail-aware P99.87 = 10.6584**, closing a substantial fraction of the shape gap.
- **Important methodological fix applied mid-stage:** initial "closure %" was computed against a single noisy MC seed, making the metric swing 54–62% depending on which seed was the denominator. Fixed by reporting **absolute gap** (mean 0.089 ± 0.009 across 3 seeds) as the primary metric and **pooled-MC closure = 57.2%** as the headline secondary metric, rather than a seed-42-only figure.
- Runtime: ~140ms (real but modest ~1.4× speedup vs MC, much slower than Clark's 566× — explicitly not conflated with Clark's number in reporting).
- A JSON export bug (loop variable shadowing `mc_p99_87`, corrupting `comparison.monte_carlo.p99_87` to silently show seed 999 instead of seed 42) was found and fixed; markdown report numbers were unaffected.

### Stage 6A — GNN training data generation ✅ Complete, went through 4 rounds of bug-fixing
Generates a dataset of (random DAG topology → MC-labeled delay mean/std) pairs for training a GNN surrogate on **arbitrary** timing graphs, not just the fixed 6-gate DAG.

**Bugs found and fixed, in order (each independently verified against raw files, not just re-reading the fix):**
1. **Vth/Vdd singularity** — unclipped Vth sampling let `(Vdd−Vth)→0` in tail draws, causing catastrophic delay blowups (mean_delay up to 25,313 vs physically plausible ~14). Fixed via Pelgrom distance cap + physical Vth bounds + raised alpha-power floor (1e-6 → 0.1). Also directly confirmed by finite-difference check that the linearized delay partials (used for Stage 6C physics features) exactly match the actual nonlinear delay formula (~1e-10 relative error) — the linearization pipeline was never broken, only the raw sampling was.
2. **Undocumented graph loss** — 652/2000 generated graphs were silently discarded (chain-only, no reconvergence) with no accounting. Fixed: manifest now reports `n_generated`, `n_dataset`, and explicit `skip_reasons`.
3. **Stale summary_stats.json** — was computed on the pre-filter 2000-graph set, causing internally-inconsistent numbers (e.g., reconvergence min reported as 0 when the actual filtered dataset's min was 1). Fixed: now computed from the actual saved dataset.
4. **Thin complex-topology test coverage** — original stratified split had only 3 nrecon≥3 graphs in the test set. Fixed by widening `n_gates_range` to (6,14) and requiring `min_reconvergence=1` at generation time, boosting nrecon≥3 test coverage to 19 graphs.
5. **Sink-predecessor mismatch** — the generator's `min_reconvergence` check (any node with ≥2 predecessors) didn't match the pipeline's actual requirement (the **sink** specifically needs ≥2 predecessors). Fixed by forcing the *first* build operation to be a split-reconverge directly on source→sink, which structurally guarantees every generated graph satisfies the pipeline's real requirement. Verified directly: 0/2000 graphs fail the sink-predecessor check post-fix (discard rate went from 32.6% to 0%).
6. **Timing-accounting artifact** — `total_generation_time_s` included graph-construction + validation overhead, not just per-graph MC+physics time, making `total/mean_time_per_graph ≈ 2020` instead of exactly 2000. Fixed by moving the timer to start right before the per-graph loop; now resolves to 2000.17 (float rounding only).

**Final locked dataset:** 2000/2000 graphs valid, 0 discards. Split: 1397 train / 296 val / 307 test, stratified by reconvergence count. Label noise: mean_noise ≈0.05%, std_noise ≈0.7–0.8% (seed-to-seed, N=10k per graph). Label distributions physically plausible (mean_delay 8–29, std_delay 0.38–1.16, scaling sensibly with graph size).

> **Note:** the original locked dataset splits (1397/296/307, nrecon 2–8) were replaced by a **regenerated dataset** concurrent with the B1 fix (see Stage 6C): 2000/2000 graphs, split 1398/298/304, nrecon range 1–4. That regenerated dataset is what Stage 6B/6C (committed numbers) were trained/evaluated on, and is not graph-for-graph comparable to earlier rounds' datasets.

Per-graph data stored: graph structure (successors, coordinates, gate loads), MC labels (mean, std at N=10k), and **physics_features** (per-gate linearized delay sensitivities ∂d/∂{L,W,Vth}, and a generalized analytical-SSTA baseline via iterative Clark MAX over arbitrary topologies — `analytical_ssta_arbitrary.py`).

### Stage 6B — Vanilla DAG-GNN baseline ✅ Complete, verified
Raw-features-only baseline (load_ff, x, y — **no** physics_features) to predict graph-level (mean, std) of critical-path delay. This is the number Stage 6C must beat to justify "physics-informed" as more than a label.

**Architecture:** 3× GraphSAGE layers (PyG `GraphSAGE` wrapper, hidden_dim=64) + BatchNorm + dropout(0.15), mean-pool readout, 2-layer MLP head → 2 outputs. ~29.7k parameters. Directed edges (respects DAG topology). Adam, lr=1e-3, MSE loss, early stopping (patience=20, max 200 epochs), gradient clipping (max_norm=1.0).

**Data hygiene:** feature/target normalization computed from **train split only**; val/test never touched during training or normalization-stat computation; splits loaded directly from Stage 6A's frozen `splits.json` (no re-shuffling).

**Results — 3-seed stability run (seeds 42/123/999), independently re-verified on the regenerated dataset:**
| Metric | GNN (mean±std across seeds) | Analytical SSTA baseline | Trivial baseline (predict train mean) |
|---|---|---|---|
| Mean delay MAE | 0.6873 → 0.41 (vanilla 6B on regenerated set: mean MAE ≈ 0.41) | 0.7216 | 3.3908 |
| Mean delay rel. error | 4.04% → 2.5% (regenerated set) | 4.57% | 22.86% |
| Std delay MAE | 0.03375 | 0.09127 | 0.1233 |
| Std delay rel. error | 4.67% → 2.6% (regenerated set) | 12.06% | 18.33% |

- GNN **beats analytical SSTA on mean** (ratio ~0.95, ~5% better) in all 3/3 seeds.
- GNN **beats analytical SSTA substantially on std** (ratio ~0.37, ~2.7× better) in all 3/3 seeds — expected, since analytical SSTA's linearization+Gaussian-MAX systematically underestimates variance/tail (consistent with Stage 4/4b/5 findings).
- GNN dramatically beats the trivial baseline (non-trivial learning confirmed).
- No overfitting: val loss consistently *below* train loss (0.71–0.77 ratio) because train loss is measured with dropout active — healthy training curves.
- Inference: 0.04–0.08 ms/graph (after fixing an earlier bug where this was accidentally measuring per-*batch*, not per-graph, timing).

**Bugs found and fixed during 6B (both self-caught by the user, both verified independently):**
1. Inference timing was batch-level, not per-graph (off by ~batch_size). Fixed by dividing by `batch.num_graphs` + added a warm-up pass before timing.
2. Analytical-baseline comparison originally only used seed 0's result. Fixed to aggregate across all 3 seeds (`comparison_aggregate`, now reports `beats_analytical_{mean,std}_seeds: 3`).

**Known minor fragility (not fixed, flagged for awareness):** `test_data` used for per-graph metadata is built via a separate `GraphDataset(...).get_data()` call from the one used internally by `create_dataloaders()` to build `test_loader`. Both are deterministic and currently produce identically-ordered lists (verified empirically — nrecon breakdown matches the real split exactly), but this is two independent construction paths that could silently misalign if one is changed without the other later.

**Files:** `gnn_baseline/{dataset.py, model.py, train.py, eval.py, run_stage6b.py}`, checkpoints `checkpoints/best_model_seed{42,123,999}.pt`, results `gnn_baseline/results/vanilla_dag_gnn_results.json` (includes `environment` and `protocol` metadata for reproducibility).

### Stage 6C — Physics-informed DAG-GNN ✅ Complete — feature injection: no headroom; **architectural MAX-bias: first positive result**

> **2026-08-29 update:** this section documents the *feature-injection* arc (original design below, plus the redesigned-feature attempt and the eventual **architectural win** at the end). Feature-level injection, across four designs/tiers, never beat vanilla. The drama reversed when the *aggregation architecture* (not features) was changed: MAX-biased aggregation beats vanilla capacity-matched on 7 seeds (Δ −0.062, CI [−0.088,−0.037]) — see ADR-008 addendum #2 and the end of this section.

**Design (unchanged across all three rounds):** 3-way ablation, architecture/hyperparameters/splits/seeds frozen identical to Stage 6B, only input features vary.
- **Vanilla** = Stage 6B, locked, reused as-is.
- **Tier A** = node features widened 3→6 dims: `[load_ff, x, y]` + `[∂d/∂Vth, ∂d/∂L, ∂d/∂W]` (per-gate linearized sensitivities from `physics_features.sensitivities`).
- **Tier A+B** = Tier A's 6-dim node features **plus** 2 graph-level scalars (`analytical_ssta.sink_mean`, `sink_std`) concatenated to the pooled embedding after `global_mean_pool`, before the MLP head.

Supporting diagnostics established as mandatory: Tier A redundancy check (`corr(∂d/∂{Vth,L,W}, load_ff)` + cross-ratio identities), Tier B leakage check (`corr(analytical_ssta.sink_mean, mc_label_mean)` + OLS β gate), a no-GNN residual-MLP baseline, lockstep verification (6C's vanilla tier must exactly reproduce 6B's standalone run), and paired cluster-bootstrap CIs (clustered by graph_id over 3 seeds) for significance.

#### Round 1 — original run (later invalidated)
Tier A: null (not significant). Tier A+B: **21.1% mean-MAE improvement, CI excluded 0, reported significant.** `corr(∂Vth, load_ff)=1.00` (Tier A redundancy, correctly diagnosed). `corr(sink_mean, label)=97.7%` (Tier B leakage, correctly flagged as a caveat). No-GNN MLP baseline (0.943) far worse than Tier A+B (0.542), used to argue the GNN was contributing beyond scalar correction. **This result does not stand — see Round 2.**

#### Round 2 — critical bug found (B1): the labels were wrong
**`ssta/monte_carlo.py` indexed variation sample columns by topological position but the samples were populated in `gate_coords` key order.** For Stage 3's fixed 6-gate DAG these orders coincide (why Stage 3/4/4b/5's locked reference was never affected and reproduces exactly post-fix). For Stage 6A's **arbitrary generated DAGs**, the orders diverge in general — so gates silently received **other gates' L/W/Vth draws**, corrupting MC labels across the Stage 6A/6B/6C dataset. This is exactly the class of bug none of the prior verification passes could have caught (labels looked internally plausible — sane ranges, low seed-to-seed noise — just attached to the wrong gate's physics).

Five more bugs (B2–B6): latent `NameError` in the default-graph MC path (B2); Tier A+B's physics-feature timing silently zeroed by a bare `except KeyError: pass` swallowing a gate-name mismatch (B3); **GPU training was never actually bit-reproducible** despite settings, because scatter-add atomics aren't deterministic — root-caused via a multi-step investigation (train-loss matched but val-loss diverged at epoch 0; CPU produced a third distinct trajectory; smoking gun: a debug print showing `Device: cuda`) (B4); CWD-dependent default config paths (B5); a `preflight()` validation function defined but never called (B6).

After fixing B1 (+ B2–B6) and regenerating the dataset with deterministic CUDA execution, the **corrected** result **reversed**: Tier A +0.040 MAE (significantly *worse*), Tier A+B +0.063 MAE (significantly *worse*) vs vanilla 0.408.

#### Round 3 — hardening audit + genuine full re-run — the committed, verified result
An independent audit re-verified every Round-2 claim against raw per-graph arrays and sha256-pinned artifacts (all matched), and fixed runner-hygiene issues: a hand-edited "verified" status string (N6); smoke-test runs able to silently overwrite real artifacts (N2); a PyG `DataLoader` wrapping a plain `TensorDataset` for the no-GNN MLP (N4); an inverted smoke-mode epoch budget (N5); plus a Pelgrom effective-dimension cap present in `sampler.py` but missing from `analytical.py` (numerically inert at current parameters). All fixes independently re-verified file-by-file. The hardened runner derives its status string dynamically from computed results, writes separate smoke artifacts, and persists training histories, OLS β, and cross-ratio diagnostics.

**Final, hardened, verified numbers (3 seeds, N=304 test graphs), recomputed from committed `stage6c_results.json` and re-derived by `verify_stage6c.py`:**

| Model | Mean MAE | vs Vanilla (cluster-bootstrap CI, n=912 obs) | Mean rel. | Signif. |
|---|---|---|---|---|
| Analytical SSTA | 0.5911 | — | — | — |
| **Vanilla DAG-GNN (6B)** | **0.4082 ± 0.0055** | — | 2.46% ± 0.04pp | — |
| Tier A (node sensitivities) | 0.4478 ± 0.0220 | delta=+0.040, CI=[+0.0241, +0.0557] | 2.69% ± 0.14pp | **Yes — CI excludes 0 (worse)** |
| Tier A+B (+ graph-level analytical) | 0.4714 ± 0.0095 | delta=+0.063, CI=[+0.0147, +0.1123] | 2.97% ± 0.07pp | **Yes — CI excludes 0 (worse)** |
| No-GNN Residual MLP (sink_mean, sink_std, n_gates) | 0.6553 ± 0.0031 | — | 4.21% ± 0.01pp | — |

- Lockstep (6B standalone vs. 6C's vanilla tier): **bit-exact**, max diff 0.0 across all 3 seeds (post-B4).
- Tier A redundancy confirmed structurally: `∂d/∂Vth ÷ ∂d/∂L = 97.5`, `∂d/∂W ÷ ∂d/∂L = −0.25` (and vth/l/w identities = +1.00/+1.00/−1.00 against `load_ff`), all holding to ~14 decimal places, independently spot-checked on random graphs. This is an algebraic identity of the linearized delay formula — Tier A's three features are always `load_ff` times fixed global scalars, carrying **zero information beyond `load_ff`** by construction, for any graph.
- Tier B leakage: `corr(sink_mean, label) ≈ 97%`, OLS β = 0.9267 (within the |β−1|≤0.1 gate).
- **Headline (original design) = both physics tiers are significantly WORSE than vanilla on the verified run.** This supersedes the earlier null-framed interim result. The no-GNN MLP (0.655) is much worse than all GNN variants, showing graph structure matters — but that does not rescue the tier-vs-vanilla comparison, which is the decision-relevant one: explicit physics injection, as designed (original design), hurts. **Later (2026-08-29):** the *redesigned* tiers (ADR-008 addendum #1) reached the same "no headroom" conclusion — redesigned Tier A still sig. worse, redesigned Tier A+B and Tier B-only both ns — so feature injection never produced a win regardless of features. The eventual win was **architectural**: MAX-biased aggregation beats vanilla, capacity-matched, on 7 seeds (Δ −0.062, CI [−0.088,−0.037]) — see ADR-008 addendum #2 and the end of this section.

**Superseded interim result (kept for the record):** the first corrected re-run gave Vanilla 0.4263, Tier A 0.4340 (CI [-0.011, +0.027]), Tier A+B 0.4307 (CI [-0.045, +0.052]) — CIs crossing zero, read as a null. That interim dataset was regenerated concurrently with hardening and is not graph-for-graph identical to the final one. The committed hardened run (table above) is the authoritative outcome. **Note (important lesson):** the previous version of this file incorrectly presented the *interim null* as the "final hardened" result; it did not match the committed artifact. Always cross-check any claim against `stage6c_results.json` / `verify_stage6c.py`, not just prior prose.

**Files:** `gnn_baseline/run_stage6c.py`, `gnn_baseline/results/stage6c_results.json` (final), `result{s,_smoke,_interim,...}.json`, `verify_stage6c.py`, and the pre-hardening backups in `gnn_baseline/{results,checkpoints}_backup_pre-hardening/` (gitignored).

---

## 6. Current locked reference numbers (for quick lookup)

| Method | P99.87 | vs MC gap | Runtime | Notes |
|---|---|---|---|---|
| Monte Carlo (N=100k, seed 42) | **10.7536** | — | 200ms | Ground truth, Stage 3 |
| Naive Gaussian (MC mean+3σ) | 10.5648 | 1.88% | — | Stage 3 §5.2 |
| Clark / linearized moments (Stage 4) | 10.4454 | 2.87% | 0.35ms (566×) | |
| Clark / empirical moments (Stage 4b) | 10.5454 | 1.94% | — | isolates shape_gap |
| Tail-aware skew-normal (Stage 5) | 10.6584 | 0.83% (pooled: ~57% closure) | 137ms (~1.4×) | |

Stage 6 uses a **different** (arbitrary-topology, regenerated) dataset, so Stage 6B/6C GNN numbers are not directly comparable to the above table — they're evaluated against per-graph MC labels (N=10k) across 2000 varied DAGs, not the single locked 6-gate reference.

**Stage 6C verified headline (for quick lookup):** 
- Feature injection (no headroom): original design Vanilla 0.4082 · Tier A 0.4478 (+0.040, sig. worse) · Tier A+B 0.4714 (+0.063, sig. worse) · Analytical 0.5911 · No-GNN MLP 0.6553; **redesigned** (ADR-008 addendum #1): Tier A 0.4562 (+0.048, sig. worse) · Tier A+B 0.4010 (−0.007, ns) · Tier B-only 0.4166 (+0.009, ns).
- **Architectural win (ADR-008 addendum #2):** MAX-biased aggregation, capacity-matched (h=54), beats vanilla on 7 seeds — Δ **−0.062** CI **[−0.088,−0.037]** (7/7 negative); aggregate maxbias_cm 0.3666 vs vanilla 0.4288. First positive result in the arc; capacity-independent (~−0.06 at matched or +41% params).
See §5 and ADR-008.

---

## 7. Verification methodology established (worth preserving as a pattern)

Every stage in this project has followed the same discipline, which should continue into Stage 6C+:
1. Never trust a summary number — recompute from the actual raw file/array.
2. Cross-check every derived quantity against at least one independent formula path (e.g., Clark's formula hand-derived, then finite-differenced against the code).
3. Check multi-seed stability before treating any single result as reliable.
4. When a "improvement" number depends on a noisy denominator (e.g., closure % against a single MC seed), report the absolute-difference version as primary and flag the ratio's sensitivity explicitly.
5. When code changes are reported as fixes, re-verify against the *actual regenerated files*, not the stated diff — several real bugs (Vth singularity, sink-predecessor mismatch, timing accounting, JSON export shadowing, the Stage 6C label-misalignment bug B1) were only caught this way.
6. **A headline result can reverse after a downstream bug fix — treat "verified" as provisional until the full pipeline (not just the stage being actively worked on) has been audited.** Stage 6C's Tier A+B result went 21%-better-and-significant → both-tiers-worse-and-significant → (an *interim* null) → both-tiers-worse-and-significant (final), each following genuine fixes upstream in shared infrastructure (`monte_carlo.py`) that no single-stage verification pass was positioned to catch. Point-estimate tables read as more conclusive than paired confidence intervals — always check whether a CI includes zero before accepting a table's implied headline.
7. Point-estimate ablation tables can misrepresent significance if read without the accompanying CIs — the "S2" and "vs. vanilla" comparisons in Stage 6C measure different things (GNN vs. no-GNN-residual, and physics-tier vs. vanilla-GNN respectively) and can each look favorable in isolation while the more decision-relevant one (physics vs. vanilla) is what actually decides.
8. **Treat documentation as a first-class artifact that can desync from code.** A context file / ADR can drift out of sync with the committed results JSON (this happened here: the prior file reported a null that the committed artifact contradicts). Cross-check any doc claim against the verified artifact and `verify_stage6c.py` before relying on it.

---

## 8. Stage 6C status and next steps

**Current state:** Stage 6C is complete. **Feature-level physics injection added no headroom across four attempts:** the original Tier A/A+B (per-node linearized sensitivities / + graph-level analytical scalars) were significantly worse than vanilla; the redesigned tiers (ADR-008 addendum #1) still found no win — redesigned Tier A sig. worse, redesigned Tier A+B and Tier B-only ns. **The pivot to architecture delivered the arc's first positive result: MAX-biased aggregation beats vanilla, capacity-matched, on 7 seeds (Δ −0.062, CI [−0.088,−0.037])** (ADR-008 addendum #2). So the "physics-informed" story resolves as: explicit feature injection hurt or tied, but *physics-consistent aggregation structure* (MAX-at-reconvergence, matching SSTA's own mechanism) helps.

**Diagnosed root causes (from Stage 6C's own diagnostics):**
- Tier A's three features are *algebraically* proportional to `load_ff` alone (fixed global ratios, provably no new information for any graph) — a feature-design problem, not a training problem. Adding three collinear features only added parameters/noise → significantly worse.
- Tier B's graph-level scalars are appended only after pooling, so the GNN can't use them during message passing — and the vanilla GNN may already implicitly learn something functionally equivalent to AT-propagation through its own layers, making the explicit analytical estimate redundant with what the network already extracts.

**Do NOT** try to fix this by running more seeds — the CIs exclude 0 in the *detrimental* direction; more seeds would narrow the CIs and more likely *confirm* the worse-than-vanilla result rather than reverse it.

**Attempts — all resolved:**
1. ~~Redesign Tier A~~ — DONE (ADR-008 addendum #1): `var_d/load_ff²` (R² vs load_ff,x,y = 0.088, non-redundant). Result: still **sig. worse** (+0.048, CI [+0.019,+0.078]) — non-redundant ≠ useful.
2. ~~Redesign Tier B as per-node~~ — DONE (ADR-008 addendum #1): per-node `AT_mean`/`AT_var` through message passing. Result: redesigned Tier A+B **ns** (−0.007, CI [−0.047,+0.032]); Tier B-only **ns** (+0.009, CI [−0.027,+0.043]).
3. ~~Architectural physics constraint~~ — DONE, **the win** (ADR-008 addendum #2): MAX-biased aggregation (`aggr=["mean","max"]`), capacity-matched (h=54), beats vanilla on 7 seeds — Δ −0.062, CI [−0.088,−0.037], 7/7 seeds negative. First positive result; capacity-independent (~−0.06 at matched or +41% params).
4. ~~Vanilla-minus-coordinates diagnostic~~ — DONE (3 seeds): dropping x,y degrades vanilla ~3× (Δ +0.83, CI [+0.72,+0.95]); vanilla does NOT reconstruct structure from `load_ff` alone — headroom was in aggregation, not injected scalars.

**Hygiene — resolved in the 2026-08-31 cleanup:** the `run_stage6c.py` "both tiers not-significant" console branch (now data-driven, deriving its summary from `significance_results`), and the two stale null-framing postmortems ([label-misalignment](docs/postmortems/postmortem-stage6c-label-misalignment-and-verified-rerun.md) and the §4/§5 of [runner-hardening](docs/postmortems/postmortem-stage6c-runner-hardening.md)) now carry forward-pointers to the authoritative feature-injection verdict + the architectural result. The stale `std MAE 0.0337 (4.67%)` figure was also scoped/corrected to 0.0188 (2.56%) in ADR-007.

**Stage 6C is resolved (real physics-informed win via architecture). Next:** Stage 7 (conformal calibration) can proceed against the MAX-bias capacity-matched model as the strongest verified backbone; Stage 8 (final comparison table across all methods — MC / analytical / Clark / tail-aware / vanilla-GNN / MAX-bias GNN / calibrated — with honest amortized data-generation cost accounting per §2's rules, and an explicit accounting of the Stage 6C bug-fix + feature-injection-null arc as a methods-section footnote, not just a results table).

---

## 9. Physics-feature redesign ideas (logged)

Ideas for a meaningful Tier-A/tier-B redesign that are *not* collinear with `load_ff`:
- `var_d / load_ff²` per-gate normalized delay variance — **IMPLEMENTED, TESTED** (ADR-008 addendum #1): non-redundant (R²=0.088) yet still sig. worse → non-redundant ≠ useful.
- Per-node `AT_mean`/`AT_var` from `analytical_ssta_arbitrary` injected at the node level — **IMPLEMENTED, TESTED** (ADR-008 addendum #1): ns (wide CI) → no headroom.
- Variance-of-log-delay (works even if the linearized mean is collinear, since variance carries the Pelgrom/spatial position dependence) — untested.
- Pairwise spatial-covariance features between reconvergent branches (captures the correlation that distinguishes MAX-heavy nodes) — untested; note the architectural MAX-bias win already addresses MAX-heavy aggregation directly.
- Any redesigned feature MUST pass the redundancy check (corr vs. existing features) and the leakage check before being trusted.
