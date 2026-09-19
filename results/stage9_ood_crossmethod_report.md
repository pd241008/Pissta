# OOD-100 Cross-Method Comparison — GNN vs Analytical Under Distribution Shift

**Date:** September 19, 2026
**Data:** `data_generation/data/ood_dataset.pkl` — 100 OOD graphs (n_gates 15–25, nrecon 2–5; distribution 21/48/23/8), MC labels N=10k seed 42 (Stage 6A convention)
**Models:** the Stage 7 checkpoints (`best_model_{vanilla,maxbias_cm}_seed{42,123,999}.pt`, trained on the pissta-2k train split, frozen 6B/7 protocol) — identical weights, no retraining
**Analytical baseline:** the **validated** reimplementation (name-correct readback + capped Pelgrom moments — the `fixed_capped` variant of the 2026-09-15 baseline audit, the label-consistent configuration), applied unchanged to the OOD graphs
**Comparison:** paired per-graph **cluster bootstrap** (clustered by graph_id across the 3 GNN seeds; `RandomState(42)`, 10,000 resamples) — the exact Stage 6C methodology
**Driver:** `diagnostics/kit1_ood_export.py` · **Artifacts:** `diagnostics/out/kit1_ood_*.json`, `gnn_baseline/results/kit1_ood100_results.json`

---

## Why this experiment

Stage 7 established that conformal **coverage** collapses under this distribution shift (pooled 25–27% vs 90% nominal, attributed to exchangeability failure, not feature extrapolation). What it deliberately did *not* measure is **point-prediction accuracy** — the conformal protocol never needed per-graph MAE. This experiment closes that gap and adds the missing third leg: the analytical baseline's error on the *same shifted distribution*. The question the paper's Discussion could only speculate on — *"does the analytical method's advantage hold, grow, or shrink under the same shift that collapses GNN calibration?"* — now has a tested answer.

## Headline result

| Method | Mean MAE on the 100 OOD graphs | vs its ID (2k test) value |
|---|---|---|
| Vanilla DAG-GNN (2k-trained) | 2.5705 (per-seed 2.3228–2.7574) | ~0.43 → **6.0× worse** |
| MAX-bias-CM GNN (2k-trained) | 2.3818 (per-seed 2.3289–2.4635) | ~0.37 → **6.4× worse** |
| **Analytical SSTA (validated fixed_capped)** | **0.2111** | 0.1129 (2k ID) → **1.9× worse** |

**The ranking does not merely persist under distribution shift — it widens by an order of magnitude.** The GNN degrades ~6×; the analytical method degrades ~1.9× (and its OOD error of 0.211 is still ~2× *better* than the GNN's *in-distribution* error of 0.37).

### Paired significance test (cluster bootstrap, 100 graphs)

| Pair | Δ (GNN − analytical) | 95% CI | Win rate (analytical) | Significant? |
|---|---|---|---|---|
| vanilla vs analytical | +2.3594 | [+1.9623, +2.7881] | 99% | **Yes** |
| maxbias_cm vs analytical | +2.1706 | [+1.8112, +2.5633] | 100% | **Yes** |

Sign convention: positive Δ = analytical better. Both CIs exclude 0 by a wide margin; the analytical method wins on 99–100% of OOD graphs. For reference, the ID (2k test split) comparison is analytical 0.1129 vs maxbias_cm ≈ 0.37 — i.e. the analytical method was *already* ahead in-distribution post-correction (this is the corrected-baseline ordering from the 2026-09-15 audit, which the Stage 8 table still carries as 0.6609 from the defected stored features), and OOD amplifies that lead from ≈3.3× to ≈11×.

## Per-nrecon breakdown (n, analytical, GNN)

| nrecon | n | Analytical MAE | Vanilla MAE | MAX-bias-CM MAE |
|---|---|---|---|---|
| 2 | 21 | 0.1825 | 1.7851 | 1.7174 |
| 3 | 48 | 0.2093 | 2.1684 | 2.0399 |
| 4 | 23 | 0.2380 | 3.2848 | 2.8764 |
| 5 | 8 | 0.2198 | 4.9912 | 4.7547 |

Two observations:

1. **The analytical method is nearly flat in nrecon** (0.18 → 0.24, and *dips* at nrecon=5). Its error model — linearization + Clark MAX — degrades with graph size/complexity but has no failure cliff, because it never extrapolates: it recomputes physics per graph from first principles.
2. **The GNN error explodes monotonically with reconvergence count**, reaching ~5.0 (vanilla) at nrecon=5 — the bucket with *no training-distribution analog* (training caps at nrecon=4, and 6–14 gates). This is the same gradient the Stage 7 OOD coverage table exposed (51–56% → 34% → 22% → 0% coverage), now confirmed in point-estimate space. The two failure modes are the same phenomenon seen through different instruments: exchangeability failure shows up as miscalibrated intervals *and* as ballooning point error, and both worsen with topological depth/complexity.

## Variant sensitivity (analytical configurations, MAE vs MC labels)

| Variant | OOD MAE |
|---|---|
| **fixed_capped (validated, used above)** | **0.2111** |
| fixed_precap (D2 unfixed) | 0.2149 |
| buggy_capped (D1 present) | 1.0499 |
| buggy_precap (both defects — the stored-feature recipe) | 1.0498 |

The readback alignment (D1) matters 5× more than the Pelgrom cap (D2) on OOD topologies — consistent with the ID audit, and worth stating because the *stored* physics features in `dataset.pkl` still carry both defects. Any future corpus regeneration should fix `data_generation/analytical_ssta_arbitrary.py` (D1 still unfixed in code) before recomputing features.

## Validation notes

- **Analytical reimplementation check:** the fixed (name-correct) readback is bit-invariant to `gate_coords` ordering (20-graph permutation sample, max |diff| = 0.0) — the same invariance property verified on the 304-graph ID audit, now confirmed on OOD topologies where topo order and coords order genuinely diverge.
- **GNN inference:** normalizer = 2k train-split stats (the deployed configuration); smoke-tested forward pass on OOD-sized graphs; 100/100 graphs scored per (config, seed) = 600 records, exported in the `stage6c_results_*.json` per-graph schema (graph_id, nrecon, mc_mean, pred_mean, pred_std, mean_mae, …) in `gnn_baseline/results/kit1_ood100_results.json` → the raw predictions are available so any downstream consumer can recompute errors directly from `ood_dataset.pkl` without re-running inference.
- **No new method code:** the analytical implementation is the identical `fixed_capped` function pair validated bit-exactly against the ID corpus; it was merely re-pointed at new graphs, as intended.

## Honest caveats

- The analytical baseline knows the *true* variation model and gate coordinates by construction — it is model-consistent with the label generator, while the GNN must infer structure from 3 node features. This asymmetry is the point of the comparison (the analytical method needs no training corpus and cannot go OOD in this sense), but it also means the result does not speak to robustness against *model* misspecification — only against topology shift.
- nrecon=5 has n=8; per-bucket means there carry high variance (reported for completeness, consistent with the Stage 7 convention of flagging small-n buckets).
- MC labels are N=10k (Stage 6A convention) — the same tail-noisy grade the GNNs were trained against; analytical is compared against the same labels on equal footing.

## Paper-facing sentences this supports

1. *"Under the same distribution shift that collapses conformal coverage to 25–27%, the GNNs' point error degrades ~6× (0.37–0.43 → 2.38–2.57 MAE), while the analytical baseline degrades only ~1.9× (0.113 → 0.211) and remains ~11× more accurate than the GNNs (paired cluster-bootstrap CI excludes zero; analytical wins on 99–100% of graphs)."*
2. *"The GNN's OOD failure deepens monotonically with reconvergence count (nrecon=5: ≈5.0 MAE), mirroring the coverage gradient of §7 — exchangeability failure manifests in both interval and point-estimate quality."*
3. *"The analytical method's advantage does not erode at the training-distribution boundary — it widens; the surrogate's value proposition is per-query cost after a paid corpus, not accuracy or robustness."*

## Files

| File | Description |
|---|---|
| `diagnostics/kit1_ood_export.py` | Driver: inference + analytical + bootstrap (single command, deterministic) |
| `diagnostics/out/kit1_ood_gnn_pergraph.json` | GNN per-graph records, both backbones × 3 seeds (stage6c schema) |
| `diagnostics/out/kit1_ood_analytical_pergraph.json` | Analytical per-graph values, all four variants |
| `diagnostics/out/kit1_ood_verdict.json` | Summary + paired bootstrap + nrecon breakdown |
| `gnn_baseline/results/kit1_ood100_results.json` | Merged single artifact (predictions + analytical + verdict) |
| `deliverables/kit1_ood100/` | Handoff package: schema-verified `ood_dataset.pkl` copy + all of the above |
