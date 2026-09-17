# ADR-010: Zenodo Release Kit — Nested Datasets, Licensing, and Public vs Internal Documentation

> **Status:** Decided  
> **Date:** September 17, 2026  
> **Last updated:** September 17, 2026

## Context

Stages 1–8 left the repo with one citable dataset (the frozen Stage 6A corpus,
2,000 graphs), generator code, and a set of manifests. Releasing to Zenodo
required deciding:

1. Whether to release only the paper-exact 2k corpus or also larger companion
   datasets (5k / 10k) for follow-up work.
2. How 5k/10k relate to the frozen 2k, given a provenance finding: the 2k
   build drew each graph's **gate-load value → gate-name pairing** in Python
   set-iteration order (`graph_generator.py` load assignment), which depends
   on the build process's `PYTHONHASHSEED`. Topology, coordinates, splits,
   and the MC labels all reproduce bit-exactly from seed 42; only that
   pairing is not re-derivable (same nondeterminism class as the B1/B4 bugs
   fixed earlier).
3. Which license(s) apply when one deposit mixes code and data.
4. Which repository documents are release documentation vs internal working
   notes.

## Options Considered

**Dataset versions:**
1. **2k only** — simplest, matches the paper exactly, but caps future work.
2. **Fresh 5k/10k** — a naive fresh run reproduces topology/coords/splits for
   graphs 0–1999 but re-draws their load pairings → different MC labels → the
   companions are *not* supersets of the paper's 2k.
3. **Nested** (chosen) — graphs 0–1999 copied verbatim from the frozen 2k;
   extension graphs generated with a canonical (sorted) load pairing, which
   makes the extension fully bit-reproducible.

**Licensing:** MIT-only (data too restrictive for reuse claims), CC-BY-4.0
only (incompatible with the code's intent), or a split (chosen): **MIT for
code, CC-BY-4.0 for data**, stated in one LICENSE file plus a pointer stub.

**Documentation scope:** commit everything, or split release documentation
(ADRs, reports, READMEs) from internal narrative notes (chosen).

## Decision

- **Three dataset versions, nested:** `pissta-2k` (paper-exact, frozen
  pickle — its manifest records the hash-seed caveat), `pissta-5k`,
  `pissta-10k` (companions, nested so that graphs 0–1999 are byte-identical
  across all three), plus `pissta-ood100` (OOD evaluation companion).
  Verified by the N1 nesting check in `zenodo/scripts/verify_release.py`
  (H1–H4 integrity, R1–R3 regeneration, N1 nesting).
- **Kit lives in `zenodo/`:** `generate_pissta.py` (build/verify any version
  from its manifest contract; never emits absolute paths — outputs are
  repo-relative, external dirs render as `<external>/<name>`),
  `verify_release.py`, per-version manifests with process constants, the
  generator commit, config snapshot, real SHA256s, and toy-model/emulation
  disclaimers. `.zenodo.json` at the repo root prefills deposit metadata.
- **License split:** `LICENSE` at the **repository root** (where GitHub and
  Zenodo expect it); `zenodo/LICENSE` reduced to a pointer stub.
- **Docs scope:** ADRs (`docs/adrs/`) remain the canonical, committed
  decision record. `PROJECT_HISTORY.md`, `physics_informed_ssta_context.md`,
  and `docs/postmortems/` are working notes — **not committed** (gitignored,
  kept on local disk); historical references to them were redirected to the
  ADRs / reports.

## Consequences

- **Positive:** the paper's 2k is untouched and re-verified (all 2,000 MC
  labels recomputed bit-exactly); 5k/10k are fully reproducible from the kit;
  nesting is machine-checked, so downstream scaling studies inherit the paper
  distribution exactly; GitHub shows the license without opening `zenodo/`.
- **Trade-off:** 5k/10k graphs 0–1999 use the frozen 2k's hash-seed-dependent
  load pairing while extensions use canonical pairing — the mixed regime is
  documented in each manifest's provenance notes rather than hidden.
- **Caveat:** a fresh independent build of the *original* 2k pairing is
  impossible by construction (no recorded hash seed); regeneration claims for
  2k are therefore limited to structure/coords/splits/labels — all verified.
- **Hygiene:** future commits must not reintroduce absolute paths in
  manifests or stdout (release hardening, same spirit as the N2 smoke-rule).
