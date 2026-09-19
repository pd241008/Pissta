# Stage 4 Pipeline Audit — Fixed Six-Gate Topology (Independent Verification)

**Date:** September 19, 2026
**Scope:** the locked six-gate reference topology pipeline — the §IV analytical figures (analytical P99.87 = 10.4454) and the Stage 4b gap decomposition (total gap **0.3083** = shape **0.2083** + linearization **0.1000**)
**Method:** the same two-step discipline used for the arbitrary-corpus analytical audit (2026-09-15): direct code read for the D1/D2 defect patterns, then bit-exact revalidation by a from-scratch reimplementation
**Driver:** `diagnostics/kit3_stage4_audit.py` · **Artifacts:** `diagnostics/out/kit3_stage4_audit_*.json`

---

## Verdict

> **CLEAN — no D1/D2 defect pattern affects the fixed-topology pipeline.** The stored analytical figures and the 0.3083 gap decomposition are reproduced (headline numbers bit-exactly; all intermediates within last-ulp float tolerance) by a from-scratch, name-keyed reimplementation computed directly from the frozen `stage3_config.json`.

This closes the paper's open scope caveat: the Stage 4/4b pipeline is now **independently audited and found sound**, in contrast to the corpus pipeline (`compute_analytical_ssta_arbitrary`), where the same audit method found both defects.

## 1. Scripts and artifacts audited

| Role | File |
|---|---|
| Run A — analytical comparator (linearized moments + Clark) | `ssta/analytical_ssta.py` |
| Run B — gap decomposition (empirical moments + Clark) | `experiments/run_stage4b_hybrid.py` |
| Frozen input: config (coordinates, loads, variation params) | `foundations/stage3_config.json` |
| Locked input: N=100k seed-42 MC reference | `results/stage3_raw.npz` (+ `stage3_summary.json`) |
| Stored outputs revalidated | `results/stage4_analytical_ssta.json`, `results/stage4b_hybrid_run.json` |

## 2. D1 — order-dependent array readback (the corpus bug class)

The corpus defect (D1) was: `compute_analytical_ssta_arbitrary` lays out per-gate moment arrays in `gate_coords` key order but reads them back by *topological* index — harmless only when the two orders coincide, which they essentially never do for random DAGs (0/50 in the ID audit).

**For the fixed topology**, two facts make the pattern structurally harmless — and both are verified, not assumed:

1. **The orders coincide for this graph.** The FIFO topological order of the six-gate DAG is exactly `['G1','G2','G3','G4','G5','G6']` — identical to the `gate_coords` key order (unique source G1; deterministic tie-breaking).
2. **The stronger property holds empirically:** recomputing the full analytical pipeline under **all 720 permutations** of `gate_coords` yields bit-identical figures (max |diff| = 0.0). Any D1-class readback sensitivity would surface as a permutation that moves the result; none does.

## 3. D2 — Pelgrom / variation-model consistency

The corpus defect (D2) was: MC labels sampled with the distance-capped Pelgrom term (`min(d, 5 µm)`) while the analytical features predated the cap unification. For the fixed topology:

- **The cap is inactive:** the maximum gate distance from origin is 4.0 µm (G6) < the 5.0 µm boundary, so capped and uncapped formulas agree per gate (verified numerically, all six gates).
- **Sampler and comparator are consistent:** `variation/analytical.py`'s per-gate Vth variance decomposes exactly as inter-die² + spatial covariance + capped-Pelgrom², matching `variation/sampler.py`'s sampling formula (verified per gate to 1e-12 relative tolerance).

Neither condition that would let D2 bite exists in this pipeline.

## 4. Bit-exact revalidation

### Run A — stored analytical figures

| Quantity | Stored | Recomputed | \|diff\| |
|---|---|---|---|
| mean | 9.256366900509 | 9.256366900509 | 0.0 |
| std | 0.396336297057 | 0.396336297057 | 0.0 |
| p95 | 9.908340109168 | 9.908340109168 | 0.0 |
| p99 | 10.178245127463 | 10.178245127463 | 0.0 |
| **p99_87** | **10.445375791680** | **10.445375791680** | **0.0** |

All 16 Run A quantities (five headline + path moments + Clark + final-sum intermediates) reproduce **bit-exactly** (repr-identical). The reimplementation reads every moment array **by name** (`idx` map), so it is immune to D1 by construction — its agreement with the stored values proves the stored pipeline's readback was also effectively name-aligned for this topology.

### Run B — gap decomposition from the locked MC reference

Re-derivation from `stage3_raw.npz` (N=100k, seed 42) with the identical recipe (empirical moments, ddof=1, Clark, Φ-weighted covariance approximation):

| Quantity | Stored | Recomputed | \|diff\| |
|---|---|---|---|
| AT_G4/G5 means, stds; d_G6 mean/std | — | — | 0.0 (6/6 exact) |
| rho_AT_G4_AT_G5 | 0.384566725069 | 0.384566725069 | 3.3e-16 |
| cov_max_d6 empirical / approx | — | — | ≤6.9e-18 |
| mu_max, std_max, mu_final | — | — | 0.0 / 0.0 / 0.0 |
| std_final | 0.412809378293 | 0.412809378293 | 5.6e-17 |
| **p99_87 (Run B)** | **10.545352381380** | **10.545352381380** | **0.0** |
| MC P99.87 | 10.753634218936 | 10.753634218936 | 0.0 |
| **total_gap** | **0.308258427256** | **0.308258427256** | **0.0** |
| **shape_gap** | **0.208281837556** | **0.208281837556** | **0.0** |
| **linearization_gap** | **0.099976589700** | **0.099976589700** | **0.0** |

14 of 17 quantities are bit-exact; 3 intermediates differ by ≤3.3×10⁻¹⁶ — **last-ulp floating-point reassociation** (numpy/BLAS reduction order across versions, e.g. inside `np.corrcoef`'s internal matmul), not a modeling or pipeline difference. All three headline gap numbers reproduce bit-exactly, as they are differences of exactly-round-tripped endpoints.

## 5. Carried-over caveats (pre-existing, disclosed in the stored report)

- **Run B consumes empirical moments (ddof=1)**: the "shape gap" partly reflects estimator variance rather than pure distributional shape — already flagged in `run_stage4b_hybrid.py` itself; reconfirmed, unchanged.
- **P99.87 convention**: mean + 3σ of the final Gaussian on the analytical side vs the empirical quantile of the locked N=100k sample on the MC side — the same convention that produced the stored figures.
- The Stage 5 tail-aware numbers (P99.87 gap 0.83%) sit on the same fixed topology and locked reference; they were *not* re-derived here, but they consume the same verified inputs (Stage 3 reference + config) and a different MAX approximation — a separate, optional follow-up.

## 6. Contrast with the corpus pipeline

| | Fixed 6-gate pipeline (this audit) | Arbitrary-corpus pipeline (2026-09-15 audit) |
|---|---|---|
| D1 (order-dependent readback) | **Absent in effect** (orders coincide; 720-permutation invariance) | **Present** (0/50 graphs aligned) |
| D2 (Pelgrom cap inconsistency) | **Absent in effect** (cap inactive at d ≤ 4 µm) | **Present** (43% of gates beyond cap) |
| Stored figures vs validated reimplementation | **Bit-exact** | Stored 0.6609 vs corrected 0.1129 MAE |
| Code status | Sound (`ssta/analytical_ssta.py`) | D1 still present in `data_generation/analytical_ssta_arbitrary.py` |

The reason the fixed pipeline escaped both defects is structural: a single hand-authored topology with a canonical ordering, and gate coordinates small enough that the Pelgrom cap never activates. The corpus pipeline failed precisely where the fixed one was safe — random orderings and coordinates spanning beyond the cap boundary.

## 7. Paper-facing confirmatory note

*"The Stage 4 analytical pipeline (fixed six-gate reference topology) was independently audited using the same defect-detection method that identified the D1/D2 defects in the arbitrary-corpus implementation: a direct code read for order-dependent array readback (D1) and variation-model inconsistency (D2), followed by bit-exact revalidation from a from-scratch, name-keyed reimplementation. Both defect patterns are structurally and empirically absent (readback invariance under all 720 gate-coordinate permutations; Pelgrom cap inactive at the topology's maximum 4.0 µm distance), and the stored analytical figures — including the 0.3083 P99.87 gap decomposition (0.2083 shape + 0.1000 linearization) — are reproduced exactly. The §IV figures therefore rest on a verified pipeline."*

## Files

| File | Description |
|---|---|
| `diagnostics/kit3_stage4_audit.py` | Audit driver (D1 permutation sweep, D2 consistency, bit-exact revalidation) |
| `diagnostics/out/kit3_stage4_audit_verdict.json` | Full verdict with per-quantity checks |
| `diagnostics/out/kit3_stage4_audit_report.json` | Recomputed Run A / Run B / gap values |
| `deliverables/kit3_stage4_audit/` | Self-contained handoff: audited code + stored outputs + verdict |
