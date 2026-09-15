"""
Verify the analytical Clark-MAX baseline in dataset.pkl — provenance, defect
isolation, and corrected value — on the 304-graph test split.

BACKGROUND (established 2026-09-15, see diagnostics/out/*.json):

  dataset.pkl's physics_features.analytical_ssta was baked in on 2026-08-21 by
  data_generation/run_stage6a.py -> compute_analytical_ssta_arbitrary()
  (data_generation/analytical_ssta_arbitrary.py), using the process moments of
  variation/analytical.py as of commit 2873f73 (NO Pelgrom d-cap).

  Two defects are baked into the stored values:

  D1 (index misalignment, B1-class). compute_analytical_ssta_arbitrary reads
     the delay-moment arrays -- which compute_delay_moments lays out in
     gate_coords KEY ORDER -- using TOPOLOGICAL indices:
         delay_mean = {name: delay_moments["mean_d"][idx[name]] ...}
     with idx = topological order. For generated graphs, topo order ==
     gate_coords key order in 0/50 sampled cases, so gates receive other
     gates' (mean, var, cov) entries. This is the same bug class as the MC
     label misalignment fixed in ssta/monte_carlo.py (2873f73, 2026-08-21);
     the analytical twin was never fixed.
     Demonstrated: passing gate_coords in topological order (which re-aligns
     arrays) changes the stored sink_mean per graph, and the buggy readback
     reproduces the stored features BIT-EXACTLY on all 304 test graphs.

  D2 (stale noise model). The MC labels were always sampled with the
     min(d, 5.0) Pelgrom cap (variation/sampler.py, since 5eebca2), but the
     analytical features predate the cap unification in
     variation/analytical.py (ba994f3, 2026-08-22). 43.15% of gates sit at
     d > 5 (max d = 20), so the cap is active on most of the corpus.

  Consequence: results/stage8_report.md's analytical MAE of 0.6609 measures
  the DEFECTED baseline. The corrected, model-consistent value (name-correct
  readback + capped moments) is ~0.1129, which REVERSES the GNN-vs-analytical
  ordering in the Stage 8 table (all GNN rows are 0.34-0.43).

METHOD
  The four variants below share the ORIGINAL propagation algorithm exactly
  (Clark MAX, the same rho approximation); they differ only in (a) array
  readback alignment (topological vs by-name) and (b) the process-moment
  model (pre-cap vs capped). The pre-cap model is reconstructed by textual
  reversion of ba994f3's single edit, matching git show 2873f73.

OUTPUTS (diagnostics/out/):
  analytical_baseline_matrix_pergraph.json   raw per-graph values, all variants
  analytical_baseline_verdict.json           summary + reproduction checks

Run:  python diagnostics/verify_analytical_baseline.py
"""
from __future__ import annotations

import inspect
import json
import math
import pickle
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from foundations.config_loader import load_config  # noqa: E402
from timing.graph import TimingGraph, Gate  # noqa: E402
from timing.delay import compute_delay_moments  # noqa: E402
from timing.clark_max import clark_max  # noqa: E402
import variation.analytical as current_analytical  # noqa: E402

DATA_PKL = REPO / "data_generation" / "data" / "dataset.pkl"
SPLITS = REPO / "data_generation" / "data" / "splits.json"
OUT_DIR = REPO / "diagnostics" / "out"


def build_pre_cap_moments_fn():
    """compute_process_moments with ba994f3's cap edit textually reverted
    (the 2873f73 state: vth_random_sigma uses d_um, uncapped)."""
    src = inspect.getsource(current_analytical)
    marker = "d_eff = min(d_um, 5.0)"
    if marker not in src:
        raise RuntimeError("cap line not found in current variation/analytical.py; "
                           "the textual reversion premise is broken")
    reverted = src.replace(marker, "d_eff = d_um  # PRE-CAP (2873f73 state)")
    if marker in reverted:
        raise RuntimeError("multiple cap lines; reversion ambiguous")
    ns = {}
    exec(compile(reverted, "<pre_cap_analytical>", "exec"), ns)
    return ns["compute_process_moments"]


def analytical_variant(tg: TimingGraph, timing_params, variation_params,
                       moments_fn, name_correct: bool):
    """Original propagation algorithm of compute_analytical_ssta_arbitrary;
    the ONLY deviation is readback alignment (name-correct when True).
    Returns (sink_mean, sink_var)."""
    order = tg.topological_order()
    topo_idx = {name: i for i, name in enumerate(order)}

    pm = moments_fn(variation_params)          # arrays in gate_coords key order
    pm_idx = pm["idx"]                          # name -> array index (correct)
    ridx = pm_idx if name_correct else topo_idx  # topo_idx reproduces the bug

    loads = {name: tg.gates[name].load_ff for name in tg.gates}
    dm = compute_delay_moments(timing_params, variation_params, pm, gate_loads=loads)

    dmean = {n: float(dm["mean_d"][ridx[n]]) for n in order}
    dvar = {n: float(dm["var_d"][ridx[n]]) for n in order}
    cov = lambda a, b: float(dm["cov_d"][ridx[a], ridx[b]])

    preds = {n: [] for n in order}
    for g, succs in tg.successors.items():
        for s in succs:
            preds[s].append(g)

    AT_mean, AT_var = {}, {}
    for name in order:
        ps = preds[name]
        if not ps:
            AT_mean[name], AT_var[name] = dmean[name], dvar[name]
        else:
            mu = AT_mean[ps[0]] + dmean[name]
            var = AT_var[ps[0]] + dvar[name] + 2.0 * cov(ps[0], name)
            for p in ps[1:]:
                mu_p = AT_mean[p] + dmean[name]
                var_p = AT_var[p] + dvar[name] + 2.0 * cov(p, name)
                cov_cp = 0.0
                for q in ps[: ps.index(p) + 1]:
                    cov_cp += cov(q, name)
                rho = cov_cp / (math.sqrt(max(var, 1e-18)) * math.sqrt(max(var_p, 1e-18)))
                rho = max(-1.0, min(1.0, rho))
                mu, var = clark_max(mu, var, mu_p, var_p, rho)
            AT_mean[name], AT_var[name] = mu, var

    sink = tg.sinks()[0]
    return AT_mean[sink], AT_var[sink]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    with open(SPLITS) as f:
        splits = json.load(f)
    with open(DATA_PKL, "rb") as f:
        dataset = pickle.load(f)

    test_ids = splits["test"]
    assert len(test_ids) == 304, f"expected 304 test graphs, got {len(test_ids)}"

    pre_cap_fn = build_pre_cap_moments_fn()
    capped_fn = current_analytical.compute_process_moments

    variants = {f"{r}_{m}": [] for r in ("buggy", "fixed") for m in ("precap", "capped")}
    per_graph = {}
    invariance_maxdiff = 0.0
    stored_repro_maxdiff = 0.0

    for gid in test_ids:
        entry = dataset[gid]
        g = entry["graph"]
        tg = TimingGraph(
            gates={n: Gate(name=n, load_ff=gd["load_ff"], x=gd["x"], y=gd["y"])
                   for n, gd in g["gates"].items()},
            successors=g["successors"],
        )
        topo = tg.topological_order()
        loads = {n: gd["load_ff"] for n, gd in g["gates"].items()}
        # gate_coords in the graph's own dict order: the state run_stage6a used
        gp = replace(cfg.variation_params,
                     gate_coords={n: g["coordinates"][n] for n in g["gates"]})
        gt = replace(cfg.timing_params, gate_loads=loads)
        mc = entry["mc_labels"]["mean"]
        stored = entry["physics_features"]["analytical_ssta"]

        row = {"mc_mean": mc, "stored_sink_mean": stored["sink_mean"]}

        for m_label, m_fn in (("precap", pre_cap_fn), ("capped", capped_fn)):
            v_bug, _ = analytical_variant(tg, gt, gp, m_fn, name_correct=False)
            v_fix, _ = analytical_variant(tg, gt, gp, m_fn, name_correct=True)
            # fixed readback must not depend on gate_coords ordering
            gp_topo = replace(cfg.variation_params,
                              gate_coords={n: g["coordinates"][n] for n in topo})
            v_fix2, _ = analytical_variant(tg, gt, gp_topo, m_fn, name_correct=True)
            invariance_maxdiff = max(invariance_maxdiff, abs(v_fix - v_fix2))

            variants[f"buggy_{m_label}"].append(abs(v_bug - mc))
            variants[f"fixed_{m_label}"].append(abs(v_fix - mc))
            row[f"buggy_{m_label}_sink_mean"] = v_bug
            row[f"fixed_{m_label}_sink_mean"] = v_fix
            row[f"fixed_{m_label}_mae"] = abs(v_fix - mc)

        stored_repro_maxdiff = max(
            stored_repro_maxdiff,
            abs(row["buggy_precap_sink_mean"] - row["stored_sink_mean"]))
        per_graph[gid] = row

    def summ(key):
        v = variants[key]
        return {"mean_mae": statistics.mean(v),
                "std_of_pergraph_mae": statistics.stdev(v)}

    verdict = {
        "n_test": len(test_ids),
        "checks": {
            "stored_features_reproduced_bit_exactly_by_buggy_precap":
                stored_repro_maxdiff == 0.0,
            "stored_vs_buggy_precap_max_absdiff": stored_repro_maxdiff,
            "fixed_readback_invariant_to_gate_coords_ordering":
                invariance_maxdiff == 0.0,
            "fixed_readback_ordering_invariance_max_absdiff": invariance_maxdiff,
        },
        "variants": {
            "buggy_precap": {
                **summ("buggy_precap"),
                "role": "reproduces the stored analytical_ssta bit-exactly; "
                        "= stage8_report.md's 0.6609",
            },
            "buggy_capped": {
                **summ("buggy_capped"),
                "role": "defect D1 only (D2 fixed); sensitivity row",
            },
            "fixed_precap": {
                **summ("fixed_precap"),
                "role": "defect D2 only (D1 fixed); sensitivity row",
            },
            "fixed_capped": {
                **summ("fixed_capped"),
                "role": "CORRECTED baseline: name-correct readback + capped "
                        "moments, consistent with the MC label sampler; the "
                        "number the paper should carry",
            },
        },
        "conclusion": (
            "The stored analytical_ssta (0.6609 MAE in results/stage8_report.md) "
            "was produced by compute_analytical_ssta_arbitrary with (D1) "
            "topological-index readback of gate_coords-ordered moment arrays and "
            "(D2) pre-Pelgrom-cap process moments. The corrected, label-consistent "
            "analytical baseline is ~0.1129 MAE. This reverses the Stage 8 "
            "GNN-vs-analytical ordering: every GNN row (0.34-0.43) is WORSE than "
            "the corrected analytical baseline, and the analytical method needs no "
            "corpus or training. The underlying code bug (D1) in "
            "data_generation/analytical_ssta_arbitrary.py is still present and "
            "should be fixed separately."
        ),
    }

    with open(OUT_DIR / "analytical_baseline_matrix_pergraph.json", "w") as f:
        json.dump(per_graph, f, indent=1)
    with open(OUT_DIR / "analytical_baseline_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print("=" * 74)
    print("Reproduction / sanity checks (304 test graphs):")
    print(f"  stored == buggy(topo-index) + pre-cap moments: "
          f"max|d| = {stored_repro_maxdiff:.3e}")
    print(f"  fixed readback invariant to gate_coords ordering: "
          f"max|d| = {invariance_maxdiff:.3e}")
    print("-" * 74)
    print("Analytical baseline MAE vs MC labels, by variant:")
    print(f"  buggy topo-index readback + PRE-cap moments "
          f"(= stored, = 0.6609): {summ('buggy_precap')['mean_mae']:.6f}")
    print(f"  buggy topo-index readback + capped moments:            "
          f"{summ('buggy_capped')['mean_mae']:.6f}")
    print(f"  FIXED name-index readback + PRE-cap moments:           "
          f"{summ('fixed_precap')['mean_mae']:.6f}")
    print(f"  FIXED name-index readback + capped moments (correct):  "
          f"{summ('fixed_capped')['mean_mae']:.6f}")
    print("=" * 74)
    print(f"\nRaw outputs: {OUT_DIR / 'analytical_baseline_matrix_pergraph.json'}")
    print(f"             {OUT_DIR / 'analytical_baseline_verdict.json'}")


if __name__ == "__main__":
    main()
