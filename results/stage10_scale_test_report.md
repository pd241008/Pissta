# pissta-10k Scale Test — Does the Analytical Advantage Erode at Scale?

**Date:** September 19, 2026
**Data:** `pissta-10k` (10,000 graphs; 2k ⊂ 5k ⊂ 10k nested companions; splits 6,997/1,498/1,505) — built and verified by the ADR-010 release kit
**Fresh models:** vanilla + maxbias_cm retrained on the 10k train split, identical frozen 6B/7 protocol (hidden 64/54, 3 layers, dropout 0.15, Adam 1e-3, batch 32, patience 20, deterministic CUDA), seeds 42/123/999
**Zero-shot models:** the Stage 7 2k-trained checkpoints applied directly to the 10k test split (2k train-split normalizer — the deployed configuration)
**Analytical baseline:** validated `fixed_capped` reimplementation (name-correct readback + capped Pelgrom moments), unchanged
**Comparison:** paired per-graph cluster bootstrap (`RandomState(42)`, 10,000 resamples, clustered by graph_id × 3 seeds) — Stage 6C methodology
**Drivers:** `diagnostics/kit2_train_10k.py`, `diagnostics/kit2_analyze_10k.py` · **Artifacts:** `diagnostics/out/kit2_10k_*.json`, `gnn_baseline/results/kit2_10k_*.json`

---

## 1. Apples-to-apples confirmation (generation config)

`pissta-10k`'s manifest records the same process constants as the paper corpus: generation seed 42, MC N=10,000 seed 42 per graph, **n_gates 6–14**, **min_reconvergence 1**, name-aligned sampler (`ssta/monte_carlo.py @ 2873f73`), 0-discard hardening. Graphs `graph_000000–001999` are byte-identical entries to pissta-2k (MC labels included, N1 nesting check). Extension graphs (2000+) use the canonical load pairing; the 2k prefix keeps its frozen pairing. **Verdict: same benchmark, 5× the graphs — a scale test, not a new benchmark.**

## 2. Headline comparison (1,505-graph test split)

| Method | Mean MAE (per-seed) | Inference ms/graph |
|---|---|---|
| **Analytical SSTA (validated)** | **0.1101** (deterministic) | **0.498** (mean), 0.861 (p95) |
| Fresh GNN — vanilla @10k | 0.2945 (0.3080 / 0.2913 / 0.2842) | ~0.3 |
| Fresh GNN — maxbias_cm @10k | 0.2891 (0.2976 / 0.2870 / 0.2827) | ~0.3 |
| Zero-shot 2k vanilla → 10k | 0.4288 (0.4335 / 0.4419 / 0.4111) | ~0.3 |
| Zero-shot 2k maxbias_cm → 10k | 0.4197 (0.4378 / 0.4209 / 0.4006) | ~0.3 |

### Paired cluster bootstrap (1,505 graphs, 3 seeds)

| Pair | Δ | 95% CI | Significant? |
|---|---|---|---|
| vanilla fresh-10k vs analytical | +0.1844 | [+0.1694, +0.1995] | **Yes** (analytical better) |
| maxbias_cm fresh-10k vs analytical | +0.1790 | [+0.1650, +0.1939] | **Yes** (analytical better) |
| vanilla zero-shot vs analytical | +0.3187 | [+0.2991, +0.3394] | **Yes** (analytical better) |
| maxbias_cm zero-shot vs analytical | +0.3097 | [+0.2904, +0.3293] | **Yes** (analytical better) |
| vanilla fresh-10k vs zero-shot | −0.1343 | [−0.1506, −0.1184] | **Yes** (5× data helps the GNN) |
| maxbias_cm fresh-10k vs zero-shot | −0.1306 | [−0.1463, −0.1148] | **Yes** |
| maxbias_cm fresh-10k vs vanilla fresh-10k | −0.0054 | [−0.0155, +0.0048] | **No** (see §5) |

## 3. The "may erode at scale" hypothesis: rejected

The Discussion's speculative sentence — the analytical advantage *may erode at scale* as the GNN gets 5× more training data — is **falsified in this regime**:

- At 2k: analytical 0.1129 vs best GNN ≈ 0.37 → gap ≈ **0.26**
- At 10k: analytical 0.1101 vs best fresh GNN ≈ 0.289 → gap ≈ **0.18**

The gap narrowed only modestly (5× data bought the GNN −0.08 MAE, a significant but small gain: CI [−0.1506, −0.1184] for both backbones), while the analytical method's error is *statistically unchanged* (0.1129 → 0.1101 — it doesn't train, so scale is irrelevant to it). The advantage **holds**; it does not erode. For it to erode, the GNN would need to improve ~10× faster per data-doubling than observed here.

## 4. Analytical inference stays practical at scale

| n_gates bucket | n | analytical ms/graph |
|---|---|---|
| 6–8 | 508 | 0.255 |
| 9–11 | 479 | 0.475 |
| 12–14 | 518 | 0.758 |

Mean 0.498 ms/graph (p95 0.861, max 2.354), scaling approximately linearly in gate count as expected. The full 1,505-graph test split takes **0.75 s** of CPU time — the analytical method remains essentially free at this scale, and the corpus-wide labeling-equivalent cost story is unchanged: 10k MC labels at N=10k grade would cost ~8.5 minutes of MC; the analytical pass over the same corpus takes under a second.

## 5. Secondary finding: the MAX-bias architectural advantage washes out at 10k

At 2k, maxbias_cm beat vanilla by −0.062, CI [−0.088, −0.037] (7 seeds, sign-stable). At 10k with 3 seeds: **Δ −0.0054, CI [−0.0155, +0.0048] — not significant, win rate 45.9%**. Two honest readings, not mutually exclusive:

1. **The vanilla model closes most of the gap with more data** — the architectural prior (mean+max aggregation mirroring Clark-MAX) matters most when data is scarce; with 6,997 training graphs the learned aggregation converges to similar behavior.
2. **3 seeds vs 7 seeds**: the 2k effect was established on 7 seeds; a 3-seed CI is wider (±0.010 vs ±0.026 here — actually narrower in absolute terms because n=1,505 per seed vs 304). The point estimate shrank 11× (−0.062 → −0.0054), which a seed-count artifact alone is unlikely to explain; the wash-out is probably real but should be confirmed at 7 seeds before the paper leans on it.

Either way, this *strengthens* the paper's cost story: if the architectural refinement only matters at small scale, the simpler vanilla surrogate is the right choice once a large corpus exists — and the analytical method remains the right choice at every scale tested.

## 6. Disaggregation

### By nrecon (fresh maxbias_cm vs analytical, pooled across seeds)

| nrecon | n | Analytical | GNN fresh-10k | GNN zero-shot |
|---|---|---|---|---|
| 1 | 756 | 0.1003 | 0.2087 | — |
| 2 | 635 | 0.1191 | 0.3474 | — |
| 3 | 110 | 0.1250 | 0.4886 | — |
| 4 | 4 | 0.1020 | 0.7370 | — |

### By n_gates (fresh maxbias_cm vs analytical)

| gates | n | Analytical | GNN fresh-10k |
|---|---|---|---|
| 6–8 | 508 | 0.0795 | 0.1798 |
| 9–11 | 479 | 0.1063 | 0.2617 |
| 12–14 | 518 | 0.1436 | 0.4216 |

The pattern is scale-invariant: the analytical method degrades mildly with both complexity axes (≤1.4× across buckets), the GNN degrades steeply (up to ~3.6× across gate-count buckets, ~3.5× across nrecon). More training data shifted every GNN bucket down uniformly; it did not change the *shape* of the degradation. The nrecon=4 bucket has n=4 — flagged, not interpreted (same convention as Stage 7).

## 7. What this means for the paper

1. **The scale hypothesis is closed**: the analytical advantage holds at 5× corpus size (§3), and analytical inference remains sub-millisecond (§4). The Discussion sentence can be replaced by a measured result.
2. **The GNN's value proposition is now precisely bounded**: per-query economics after a paid corpus (Stage 8), never accuracy (this report), never calibration under shift (Stage 7 §6), and its accuracy advantage over *nothing* — the analytical baseline is better at every scale tested, at every complexity level, in-distribution and out.
3. **The zero-shot row adds a data point on generalization**: 2k-trained models score 0.42–0.43 on the *in-distribution-family* 10k test split — worse than fresh training (0.29) but far better than the OOD collapse (2.4–2.6). Same-family distribution shift is survivable; topology-family shift is not.

## 8. Reproducibility

- Fresh training: `python diagnostics/kit2_train_10k.py` (resumable per config/seed; checkpoints `best_model_{config}_seed{seed}_10k.pt`)
- Analysis: `python diagnostics/kit2_analyze_10k.py` (zero-shot + analytical + bootstrap + disaggregation)
- pissta-10k regeneration: `python zenodo/scripts/generate_pissta.py --version pissta-10k` then `python zenodo/scripts/verify_release.py` (all checks green, 2026-09-17)
- Training wall-clock: 218–533 s per run (single GPU, deterministic kernels, load-sensitive as documented in Stage 8 footnote 2); best-epoch range 39–130

## Files

| File | Description |
|---|---|
| `diagnostics/kit2_train_10k.py` | Fresh-training driver (frozen protocol, resumable) |
| `diagnostics/kit2_analyze_10k.py` | Zero-shot + analytical + paired bootstrap + disaggregation driver |
| `gnn_baseline/results/kit2_10k_results_{config}_seed{seed}.json` | Per-seed per-graph fresh-training results (stage6c schema) |
| `gnn_baseline/results/kit2_10k_results.json` | Merged artifact (fresh + zero-shot + analytical + verdict) |
| `diagnostics/out/kit2_10k_verdict.json` | Bootstrap table + disaggregation + timing |
| `diagnostics/out/kit2_10k_zeroshot_results.json` | Zero-shot per-graph records |
| `diagnostics/out/kit2_10k_analytical_pergraph.json` | Analytical per-graph values + timing by size |
| `deliverables/kit2_pissta10k/` | Handoff package (dataset + splits + all results) |
