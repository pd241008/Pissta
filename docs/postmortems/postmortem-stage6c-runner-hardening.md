# Postmortem — Stage 6C Runner Hardening (Audit Follow-Up)

> **Date:** August 21, 2026
> **Predecessor:** `postmortem-stage6c-label-misalignment-and-verified-rerun.md` (the bug-sweep session report, moved here from the repo root)
> **Scope:** independent audit of the post-sweep repo state, defect fixes in `gnn_baseline/run_stage6c.py` / `variation/analytical.py`, smoke validation, and a full deterministic re-run on the corrected dataset.

---

## 1. Context

The bug-sweep session (same day, earlier) had fixed six defects — most critically the Monte-Carlo label misalignment (`ssta/monte_carlo.py` indexed sample columns by topological position instead of `gate_coords` name order) — regenerated the Stage 6A dataset, and re-run Stages 6B/6C with deterministic CUDA execution. That re-run **reversed** the Stage 6C headline: vanilla DAG-GNN 0.4082 ± 0.0055 is best; Tier A (+0.040) and Tier A+B (+0.063) are significantly *worse*.

The follow-up audit verified every claim of that sweep against raw artifacts before building on it:

| Audit check | Result |
|---|---|
| All 9 config/seed aggregates recomputed from per-graph arrays | Match to float32 precision; sha256 digests pin interim/final/6B files as one run |
| Lockstep 6B vs 6C vanilla | Bit-exact: max diff 0.0, best_val identical to ~17 digits |
| Stats tail | Every block populated (Tier A diagnostics, significance CIs, leakage, B5 recount, S2 decision) |
| Full-run signature | train_time 29–150 s per training, best_epoch up to 146 |
| Param counts | 29,698 / 29,890 / 30,018 (deltas +192/+128 preserved); ResidualMLP 194 |
| Stage-3 locked reference re-executed | mean/std/P95/P99/P99.87 = 9.3065 / 0.4194 / 10.0243 / 10.3675 / 10.7536 — exact (label fix provably a no-op where gate order already matched) |
| Leakage triangulation | 0.1495 × 3.8708 = 0.5786 = no-GNN MLP MAE |
| Tier A mechanism | corr(∂Vth, load_ff) = +1.000000, ∂W = −1.000000 exactly over n=14,022 gates; magnitude identities exactly +1/+1/−1 |
| Tests | 11/11 pass |

## 2. Defects found & fixed

### N6 — hardcoded status string (audit finding)
Commit `619c36a` had edited the runner's status string to `"complete — verified end-to-end…"` *after* the artifact was written, leaving code and artifact vintages out of sync (the on-disk JSON honestly still said `"implemented, pending…"`). Every claim in the new string was factually true of the artifact, but the pattern asserts outcomes unconditionally — a failing or smoke re-run would inherit them.
**Fix:** status is now assembled at runtime from computed blocks (lockstep result, CI computation, physics timing, S2 verdict, convergence gate, B5 persistence).

### N2 — final save not smoke-gated
A completed smoke run would have overwritten `stage6c_results.json`, indistinguishable from a full run and inheriting its claims.
**Fix:** smoke mode writes `stage6c_results_smoke.json` / `stage6c_results_interim_smoke.json` and checkpoints into `checkpoints_smoke/`; status is tagged `SMOKE run | …`. Smoke can no longer touch full-run artifacts.

### N4 — PyG DataLoader over TensorDataset
The no-GNN MLP wrapped a plain `TensorDataset` in `torch_geometric.loader.DataLoader`. Worked on the installed PyG 2.8; latent portability hazard.
**Fix:** `torch.utils.data.DataLoader`.

### N5 — inverted smoke ternary for the MLP
`200 if SMOKE else 1000` gave the MLP 20× the GNNs' smoke epoch budget.
**Fix:** `50 if SMOKE else 1000`.

### Residues closed
- **β≈1 slope gate (was C5):** OLS coefficient of normalized `sink_mean` persisted in `ols_floor` / `convergence_gates` / `s2_convergence_decision`; gate |β−1| ≤ 0.1.
- **sW/sL cross-ratio (was C7):** persisted alongside sV/sL; expected −L_nom/(2·W_nom).
- **k-placement honesty:** identities persist `k_value` and whether the test discriminates placement at all (it does not at k=1.0 — reported, not tuned).
- **vdd dual-sourcing (was C7):** `preflight()` and the magnitude identities now read `timing_params.vdd_v` — the same source `delay_partials` consumes — instead of the unused `variation_params.vdd_v` default.
- **Training histories** persisted in the final results JSON (previously only in interim saves).

### Pelgrom effective-dimension cap unified
`variation/sampler.py` capped d at 5.0; `variation/analytical.py` did not. The analytical side now applies the same `min(d_um, 5.0)`. Verified numerically inert on current parameters (max d ≈ 4.12 < 5): pre/post process moments identical.

### Owed tooling
`verify_stage6c.py` committed to the repo root: recomputes all aggregates + sha256 digests from per-graph arrays without bulk transmission; accepts an artifact path argument (works on interim/smoke files).

## 3. Validation

- Smoke run (`VLSI_SMOKE=1`) exits 0 end-to-end; derived status correctly reads `"SMOKE run | lockstep=mismatch (max_mean_diff=0.298) | …"` — under the old hardcoded string this exact run would have claimed "verified".
- md5 of all 11 protected files (full results JSONs + 9 checkpoints) byte-identical after the smoke run.
- New diagnostics live in smoke stdout: cross-ratio vth/l = 97.5000 ± 1.2e-14 (expected α·L_nom/(Vdd−Vth)); cross-ratio w/l = −0.2500 ± 0.0 (expected −L_nom/(2·W_nom)); β = 0.9267 PASS; k-placement flagged non-discriminative at k=1.0.
- 11/11 tests pass; Stage-3 locked reference still reproduces exactly after all edits.

## 4. Full re-run on hardened code

Re-ran with `VLSI_SMOKE` unset after backing up prior artifacts. Outcome:

- **Training-neutral hardening proven:** all 9 per-graph sha256 digests identical across the hardening boundary; best epochs reproduce (vanilla 69/112/105, tier_a 115/93/126, tier_ab 70/99/76); ResidualMLP bit-identical (0.5786).
- **Derived status reads:** `full run | lockstep=exact (max_mean_diff=0.0) | cluster-bootstrap CIs computed (n=304) | physics timing tier_ab=0.42 ms/graph | vs_vanilla: tier_a=ns (delta=+0.0077, CI=[-0.0113, +0.0272]), tier_ab=ns (delta=+0.0044, CI=[-0.0450, +0.0523]) | S2 no-gnn rel=+34.4% -> gnn_beyond_scalar_residual | s2_convergence mlp_le_ols_plus_001=True | B5 reconciliation persisted`.
- **New fields persisted:** training histories, OLS slopes (β_mean = 1.0119, gate PASS), sW/sL cross-ratio (−0.2500 vs expected −L_nom/(2·W_nom) = −0.25), k-placement flag (non-discriminative at k=1.0).
- Aggregates (this pre-hardening re-run, internal to this audit): vanilla **0.4263 ± 0.0063**, tier_a 0.4340 ± 0.0299, tier_ab 0.4307 ± 0.0147; lockstep exact (max diff 0.0). **This is the superseded interim re-run** — see §1 Context and the note below for the authoritative post-hardening numbers (vanilla 0.4082; Tier A +0.040; Tier A+B +0.063, both sig. worse).

## 5. Standing conclusions (hardened final result)

**Note on §4 vs §1 Context:** §4 above quotes the *interim* re-run aggregates (vanilla 0.4263, Tier A +0.0077, Tier A+B +0.0044) which predate the final hardened run. The authoritative outcome, verified against the committed artifact at 2026-08-28 (per-graph sha256 digests, lockstep max diff 0.0, cross-checked by `verify_stage6c.py`), is the **sig.-worse result** stated in §1 Context. The table below is the authoritative final row:

| Model | Mean MAE | vs Vanilla |
|---|---|---|
| Vanilla DAG-GNN (6B) | **0.4082 ± 0.0055** | — |
| Tier A (node sensitivities) | 0.4478 ± 0.0220 | +0.040, CI [+0.0241, +0.0557] — **significantly worse** |
| Tier A+B (+ graph-level analytical) | 0.4714 ± 0.0095 | +0.063, CI [+0.0147, +0.1123] — **significantly worse** |
| Analytical SSTA | 0.5911 | identity floor |
| OLS floor / ResidualMLP | 0.6494 / 0.6553 | non-graph methods plateau above all GNNs |

S2 verdict: rel(MLP vs Tier A+B) = +34.4% → graph structure contributes far beyond scalar residual correction. Durable findings unchanged: Tier A's structural redundancy is exact (±1.000000 correlations); analytical sink_mean remains ~97.6% correlated with MC mean (honest-leakage framing). **The 0.4263/0.4340/0.4307 numbers recorded in §4 are superseded by the hardened run above and kept only as the historical interim step.**
