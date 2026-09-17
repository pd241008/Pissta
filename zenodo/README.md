# Pissta Dataset Release Kit (Zenodo)

Datasets of **random timing DAGs with Monte-Carlo-labeled critical-path delay
statistics** (mean/std) under correlated process variation, produced by the
[Pissta](https://github.com/Avinash-634/Pissta) physics-informed SSTA framework.

Each graph is a single-source/single-sink DAG (6–14 gates, 1–4 reconvergence
points, 2D spatial coordinates, per-gate loads) with:

- `mc_labels` — critical-path delay mean/std from a vectorized Monte Carlo over
  correlated variation (inter-die + spatial + Pelgrom mismatch), **N=10,000
  samples, seed 42** per graph (Stage 6A convention — a tail-noisy grade,
  distinct from the locked N=100k Stage 3 reference);
- `physics_features` — per-gate linearized delay sensitivities ∂d/∂{L,W,Vth},
  per-gate `var_d/load_ff²`, and an analytical-SSTA baseline (iterative Clark
  MAX) per graph;
- a stratified 70/15/15 train/val/test split by graph (no graph crosses splits).

> **⚠️ Emulation disclaimers (apply to every version)**
> All numerical parameters (α=1.3, Vdd=1.0 V, L_nom=45 nm, W_nom=90 nm,
> Vth_nom=0.40 V) are deliberately simple demonstration parameters for
> software/methodology validation — **not silicon-calibrated**, and never to be
> presented as real technology results. Labels are Monte-Carlo **emulations**
> from the repository's own variation sampler, not measurements or SPICE
> simulations of a real process.

---

## Dataset versions

| Version | Graphs | Split (train/val/test) | Size | Role |
|---|---|---|---|---|
| **pissta-2k** | 2,000 | 1398 / 298 / 304 | 4.5 MB | **Paper-exact** — the Stage 6A dataset used in every reported Stage 6B/6C/7/8 result |
| **pissta-5k** | 5,000 | 3499 / 748 / 753 | 12 MB | Companion — 2k ⊂ 5k |
| **pissta-10k** | 10,000 | 6997 / 1498 / 1505 | 24 MB | Companion — 2k ⊂ 5k ⊂ 10k |
| **pissta-ood100** | 100 | evaluation-only | 203 KB | Companion — OOD topology eval (15–25 gates, ≥2 reconvergences; Stage 7 Step 6) |

**Paper-exact vs companion.** `pissta-2k` is the *frozen artifact* the paper's
numbers were computed on — it is published verbatim and re-verified, never
rebuilt. `pissta-5k`/`pissta-10k` are *companion scaling* releases built around
it: graphs `graph_000000`–`graph_001999` are **byte-identical entries (MC
labels included) copied from pissta-2k**; graphs `graph_002000+` are extension
graphs. Any model trained on pissta-2k consumes an identical prefix of the
larger versions.

**Nesting guarantee** `pissta-2k ⊂ pissta-5k ⊂ pissta-10k` is enforced as a
verification check (`verify_release.py`, N1) over all 2,000 prefix entries.
Splits are regenerated per version by the frozen stratified procedure and
therefore differ across versions by design; graph sets do not.

**Label quality** (seed-to-seed, N=10k): mean-delay noise ≈0.05–0.06%, std
noise ≈0.7–1.1%. Dataset-level statistics (mean delay ≈15.96–16.22 ± ≈3.8;
per-graph std ≈0.73–0.74) are in each `summary_stats.json`.

---

## What is (and is not) bit-reproducible

Verified 2026-09-17 against repo commit `d697021` (details in each
`manifest.json` and in [CODE_README.md](./CODE_README.md)):

| Quantity | Status |
|---|---|
| Graph topology, coordinates, gate-count/nrecon distributions | ✅ bit-exact from seed 42, current code |
| Stratified train/val/test split | ✅ bit-exact (seed 42) |
| **MC labels (the supervised targets)** | ✅ **bit-exact** (N=10k, seed 42, name-aligned sampler) |
| Gate-load *values* per graph | ✅ bit-exact multiset |
| Gate-load *pairing* (value → gate name) in pissta-2k | 🔒 frozen artifact — the 2026-08-17 build drew loads in Python set-iteration order (`PYTHONHASHSEED`-dependent) |
| Gate-load pairing in extension graphs (2000+) | ✅ canonical rule: ascending load value → lexicographic gate name |

One legacy caveat: the 2k prefix's `physics_features` were computed
2026-08-21 with pre-2026-08-22 process moments (`variation/analytical.py`
changed the next day). MC labels are unaffected and bit-consistent across all
versions; for physics-feature work on the prefix, see the provenance note in
the 5k/10k manifests.

---

## Layout of this kit

```text
zenodo/
├── README.md            ← you are here
├── CODE_README.md       ← how to regenerate + verify + upload
├── LICENSE              ← MIT (code) + CC-BY-4.0 (data)
├── requirements.txt     ← exact package pins
├── environment.yml      ← conda equivalent
├── scripts/
│   ├── generate_pissta.py   # build/verify a dataset version from its manifest
│   └── verify_release.py    # hash/schema/count/label/nesting verification
└── data/                ← generated (gitignored): one folder per version
    └── pissta-*/        #   dataset.pkl, splits.json, summary_stats.json,
                         #   manifest.json, VERIFICATION.txt
```

Each `manifest.json` records: process constants (seeds, MC sample count, gate
ranges, split fractions), the generator commit hash, a config snapshot, SHA256
+ size of every published file, the disclaimers above, and per-version
provenance notes.

## Licenses

Code: **MIT**. Data: **CC-BY-4.0**. See [LICENSE](./LICENSE).

## Citation

Cite the Zenodo dataset deposits (DOIs are minted on first upload; reserve the
concept DOI and record it in `.zenodo.json` at the repo root). Until then,
cite the repository: https://github.com/Avinash-634/Pissta
