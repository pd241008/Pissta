# Stage 8 — Final Cross-Method Comparison Table

**Status:** assembled from real measured numbers only. All per-graph MAE numbers were
recomputed from the raw `per_graph` arrays (see Verification Appendix). GNN rows are the
reruns performed for instrumentation (same hyperparameters/seeds as the committed artifacts;
bit-reproducible — max per-graph diff 0.0 vs the committed `stage6c_results_maxbias*.json`).

**Benchmark:** per-graph error over the 304-graph test split of the current locked dataset
(1398/298/304, nrecon 1–4), evaluated against the per-graph MC labels (N=10k). This is the
Stage 6 benchmark — the single fixed 6-gate DAG of Stages 3–5 is *not* this table's
accuracy basis (MC/analytical/tail-aware rows that were only measured there are flagged).

## Table

| Method | Mean MAE (toy units) | Inference cost (ms/graph) | Training cost (s) | Data-generation cost (amortized) |
|---|---|---|---|---|
| Monte Carlo (ground truth, N=10k labels) | 0 (self-reference) | 51.3 (**labels**, N=10k); 200 (N=100k, Stage-3 tail grade) | 0 | — (it is the generator; cost shown under inference) |
| Analytical SSTA (Clark, arbitrary-DAG) | **0.6609** (4.21% rel) | **0.563** (measured, 304 graphs); 0.35 (6-gate DAG, Stage 4) | 0 | 0 |
| Tail-aware skew-normal (Stage 5) | **n/a** on this benchmark — implemented only for the fixed 6-gate DAG (P99.87 gap 0.83%) | 137 (6-gate DAG, Stage 5) | 0 | 0 |
| Vanilla DAG-GNN | 0.4288 (7 seeds); 0.4082 (3-seed cadence) | 0.22–0.35 (batch-eval, load-sensitive) | 106–154 (avg per run; load-sensitive) | 102.5 s total (= 51.3 ms/graph × **2,000** graphs, shared) |
| MAX-bias GNN (capacity-matched, h=54) | **0.3666** (7 seeds); Δ vs vanilla **−0.062**, CI **[−0.088, −0.037]** | 0.28 | 112 (avg) | same shared corpus (not re-incurred) |
| MAX-bias GNN (full-cap h=64, +41% params) | **0.3438** (3 seeds); Δ **−0.064**, CI **[−0.089, −0.040]** | 0.28 | 120 (avg) | same shared corpus (not re-incurred) |
| Combined: MAX-bias-CM + redesigned Tier A+B | **0.3607** (7 seeds); Δ vs vanilla **−0.068**, CI **[−0.105, −0.032]**; Δ vs MAX-bias-CM alone **−0.006**, CI **[−0.033, +0.022]** (ns) | **0.81** (0.25 GNN + 0.56 per-node physics features) | 103 (avg) | same shared corpus (not re-incurred) |

### Footnotes

1. **All GNN rows** are trained on the same 2,000-graph corpus (1398 train / 298 val / 304 test).
   One-time generation = 102.5 s of MC labeling at N=10k (**51.3 ms/graph**)
   — `data_generation/data/manifest.json`. The training cost is **additional** to this corpus
   cost and is re-incurred per variant.
2. **Training-time sensitivity.** Wall-clock seconds on a single GPU with deterministic CUDA.
   Bit-reproducible *in the metric* (per-graph lockstep diff 0.0), but the *wall-clock* is
   load-sensitive: the same vanilla run measured 106–154 s across three separate invocations
   (±35%), and inference measured 0.13–1.01 ms/graph. Report to one digit, and as a range.
3. **Analytical accuracy.** 0.6609 is the current, directly-recomputed number
   (`dataset.pkl` + `splits.json`, |ana_sink_mean − mc_mean| averaged over the 304 test
   graphs = 0.660890). The 0.5911 figure in older docs is **stale** (an earlier dataset
   revision) — do not carry it into the paper.
4. **Tail-aware row is not comparable to the others** in the accuracy column: the
   skew-normal 3-moment MAX was never implemented for arbitrary topologies. Its Stage 5
   numbers (P99.87 gap 0.83% on the fixed 6-gate DAG, ~57% shape-gap closure, 137 ms) are
   included for completeness and flagged, not merged into this benchmark.
5. **MC "accuracy" of 0** is by construction (a method evaluated against its own labels).
   The GNN rows are surrogate approximations and are *never* more accurate than the labels
   they were trained on — the selling point is per-query economics + generalization.

## Amortized-cost paragraph (what it honestly shows)

**The per-query game is overwhelmingly won; the lifecycle game is not.** A surrogate
forward pass costs 0.22–0.81 ms/graph against 51.3 ms/graph for an N=10k MC label
(~60–230×) and 200 ms/graph for the N=100k tail-grade MC label of Stage 3 — the regime the
research question actually cares about (~250–900×). That is real, and it survives scrutiny
*provided the model already exists*. But every GNN row sits on top of an up-front bill that
is **not** negligible: the 2,000-graph MC labeling corpus costs 102.5 s, and training adds
~103–154 s per variant. Total ~210–260 s before the first query. The break-even against
"just run MC per query" is roughly **4,000 graphs** at N=10k grade, or **~1,100 graphs** at
N=100k grade. For a single evaluation, or a study below a few thousand queries, MC at N=10k
(51 ms/graph) is cheaper end-to-end than training a surrogate — this must be said plainly.
The "much-lower-cost-than-MC" framing is therefore load-bearing in the **amortized / on-line
regime** (many queries per trained model), and is not a per-lifecycle speedup.

**The training-cost premium partially undercuts the framing, and should be said so.** The
~110 s of training on top of the 102.5 s of data generation roughly *doubles* the MC
labeling bill rather than being a rounding error, and the MAX-bias variant bears no less of
it than vanilla (112 s vs 123 s — capacity-matched; full-cap 120 s). If the paper's comparison
stops at inference-time ms/graph, it hides that the nominal "winner" only exists after a paid
MC-corpus + training phase. The honest sentence is: *the surrogate is cheaper per query by
two orders of magnitude once trained, but its total cost-of-first-query is dominated by —
and roughly double — the MC data it was trained to imitate.* That tension is the real cost
model of the method, not an edge case.

**Accuracy is not a place where the surrogate "wins".** All GNN rows are strictly less
accurate than the labels they approximate (0.367–0.429 MAE vs 0 for MC-by-construction),
and no variant closes the gap to the analytical baseline's 0.661 by more than the
architectural grouping does. The MAX-bias improvement (−0.062 to −0.068, CIs entirely below
0, sign-stable 7/7 and 3/3) is a *surrogate-vs-surrogate* win: it makes the cheap
approximator better, it does not make it exact.

**The combined variant is not worth taking.** Adding the redesigned per-node Tier A+B
physics features to MAX-bias-CM (0.3607 vs 0.3666) buys a Δ of −0.006 that is statistically
indistinguishable from zero (CI [−0.033, +0.022], per-seed deltas sign-unstable), at the cost
of a 0.56 ms/graph physics-feature computation at inference (0.81 vs 0.28 ms total). This
reconfirms the Stage 6C arc from the other direction: on top of the working architecture,
**feature-level physics injection still adds nothing**.

### Calibration caveat (carried from Stage 7, applies to the selected backbone)

The split-conformal evaluation (ADR-009) shows both backbones under-cover the MAX-heavy
nrecon=2 regime (pooled 90% nominal: vanilla 0.895 / maxbias_cm 0.908 overall, but
0.84–0.87 at nrecon=2), and the tighter intervals of maxbias are what cause its
under-coverage there. Point-estimate MAE does not imply calibration; the paper must report
per-bucket coverage, not just the pooled number. No coverage claim is made under
distribution shift (OOD is a documented follow-up, not a result).

## Verification Appendix (independent re-derivation, 2026-09-13)

Fresh code in `/tmp/opencode/verify_stage8.py` (does **not** call the project's
`paired_cluster_bootstrap_ci` / `_agg`); every number below recomputed from raw per-graph arrays:

- **Point estimates.** Recomputed mean/std MAE from `per_graph` arrays match each artifact's
  `stability_summary` to 6 decimals for all 6 configs (vanilla/maxbias_cm × 7s,
  vanilla/maxbias × 3s, vanilla/maxbias_cm_tierab × 7s). No drift.
- **Lockstep vs committed artifacts.** Re-run vanilla *and* variant per-graph MAE arrays vs
  the backed-up committed artifacts (`/tmp/opencode/artifacts_backup`): **max |d| = 0.0** on
  all shared seeds (CM 7/7, FC 3/3) — the re-runs are the same experiment, instrumented with
  `train_time` only.
- **CIs re-derived independently** (clustered per graph, averaged over seeds, 10 000
  resamples, fresh RNG seed 7 vs the harness's 42):

  | Pair | Harness CI | Independent CI |
  |---|---|---|
  | maxbias_cm − vanilla (7s) | [−0.0880, −0.0369] | [−0.0886, −0.0372] |
  | maxbias − vanilla (3s) | [−0.0889, −0.0396] | [−0.0893, −0.0398] |
  | maxbias_cm_tierab − vanilla (7s) | [−0.1041, −0.0315] | [−0.1048, −0.0324] |
  | maxbias_cm_tierab − maxbias_cm (7s) | — (not in harness) | [−0.0330, +0.0218] (ns) |

  All "vs vanilla" intervals exclude 0 → the two MAX-bias and the combined rows are
  significantly *better* than vanilla under both derivations. The combined-vs-maxbias_cm
  interval includes 0 → not distinguishable.
- **Combined vs MAX-bias-CM per-seed deltas** (fresh bootstrap): +0.037, −0.016, −0.062,
  +0.023, −0.027, +0.009, −0.003 — sign-unstable, consistent with the null.
- **Analytical SSTA MAE** recomputed directly from `dataset.pkl` + `splits.json`:
  0.660890 (mean), 0.112250 (std) — matches every fresh and committed artifact.
- **Analytical per-graph runtime** measured this session: 0.563 ms/graph over the 304 test
  graphs (warm, after 8-graph warmup).

## Artifacts

| File | Contents |
|---|---|
| `gnn_baseline/results/stage6c_results_maxbias_cm.json` | Re-run (7 seeds), now incl. `train_time`, `history`, `avg_train_time_s` |
| `gnn_baseline/results/stage6c_results_maxbias.json` | Re-run (3 seeds), same additions |
| `gnn_baseline/results/stage6c_results_maxbias_cm_tierab.json` | **New** combined variant (7 seeds) |
| `gnn_baseline/run_maxbias_tierab.py` | New combined-ablation harness |
| `gnn_baseline/{run_maxbias,run_nocoor,run_tier_b_only}.py` | Fixed: `train_time` no longer silently dropped (`avg_train_time_s` added to summaries) |
| `/tmp/opencode/artifacts_backup/` and `gnn_baseline/results_backup_pre-traintime/` | Pre-re-run committed artifacts |