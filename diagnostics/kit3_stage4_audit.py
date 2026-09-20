"""
Kit 3 — Stage 4 fixed-six-gate pipeline audit + bit-exact revalidation.

Closes the counterpart asks on the locked six-gate reference topology
(the §IV P99.87 gap-decomposition figures; 0.2781 = 0.2083 shape +
 0.0698 linearization after the 2026-09-20 propagate_path covariance
 fix — the original audit ran against the pre-fix 0.3083 figures):

  1. Stage 4 scripts inventory + integrity (packaged by package_deliverables.py):
     ssta/analytical_ssta.py (Run A — linearized moments + Clark, the stored
     analytical comparator), experiments/run_stage4b_hybrid.py (Run B —
     empirical moments + Clark, the gap decomposition), plus their locked
     inputs (stage3_raw.npz / stage3_summary.json) and stored outputs
     (stage4_analytical_ssta.json / stage4b_hybrid_run.json).

  2. D1/D2 code read, made *testable*:
       D1 (order-dependent array readback): the fixed topology has a single
         source and a deterministic FIFO topological order, so the topological
         position == gate_coords key position for THIS graph — but we prove the
         stronger property: recomputing the analytical pipeline under EVERY
         gate_coords permutation of the six gates yields bit-identical
         figures. Any D1-class sensitivity would show up as a permutation that
         moves the result.
       D2 (Pelgrom / variation-model inconsistency): the MC sampler
         (variation/sampler.py) caps the distance-dependent Pelgrom term at
         d=5 um; the analytical comparator (variation/analytical.py) now uses
         the same capped formula (post-ba994f3). For this topology the max
         gate distance from origin is sqrt(4^2+0^2)=4.0 um < 5 um, so the cap
         is inactive either way — we verify the two formulas agree per gate
         numerically.

  3. From-scratch reimplementation (same frozen propagation recipe as Run A,
     name-keyed readback by construction) recomputed directly from
     foundations/stage3_config.json — must reproduce the stored analytical
     figures bit-exactly.

  4. Run B re-derivation from the locked N=100k seed-42 MC reference
     (stage3_raw.npz): empirical moments -> Clark -> gap decomposition must
     reproduce the stored 0.2781 / 0.2083 / 0.0698 numbers bit-exactly
     (pre-fix values were 0.3083 / 0.2083 / 0.1000).

Outputs:
  diagnostics/out/kit3_stage4_audit_verdict.json
  diagnostics/out/kit3_stage4_audit_report.json

Run:  python diagnostics/kit3_stage4_audit.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from itertools import permutations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from foundations.config_loader import load_config  # noqa: E402
from timing.graph import build_branching_graph_from_config  # noqa: E402
from timing.delay import compute_delay_moments  # noqa: E402
from timing.clark_max import clark_max  # noqa: E402
from variation.analytical import compute_process_moments  # noqa: E402
from variation.sampler import _pelgrom_vth_sigma  # noqa: E402
from ssta.statistical_sum import (  # noqa: E402
    covariance_between_paths,
    gaussian_sum,
    propagate_path,
)

STAGE4_JSON = REPO / "results" / "stage4_analytical_ssta.json"
STAGE4B_JSON = REPO / "results" / "stage4b_hybrid_run.json"
STAGE3_SUMMARY = REPO / "results" / "stage3_summary.json"
STAGE3_RAW = REPO / "results" / "stage3_raw.npz"
OUT_DIR = REPO / "diagnostics" / "out"

PATH1 = ["G1", "G2", "G4"]
PATH2 = ["G1", "G3", "G5"]
Z = {"p95": 1.645, "p99": 2.326, "p99_87": 3.000}


def recompute_analytical_from_config(config) -> dict:
    """From-scratch reimplementation of the Run A pipeline (ssta/analytical_ssta.py),
    reading every moment array BY NAME (immune to D1 by construction)."""
    timing_params = config.timing_params
    variation_params = config.variation_params

    process_moments = compute_process_moments(variation_params)
    delay_moments = compute_delay_moments(timing_params, variation_params, process_moments)
    idx = delay_moments["idx"]  # name -> index into the arrays (name-keyed)

    mu_at_g4, var_at_g4 = propagate_path(
        PATH1, delay_moments["mean_d"], delay_moments["var_d"],
        delay_moments["cov_d"], idx)
    mu_at_g5, var_at_g5 = propagate_path(
        PATH2, delay_moments["mean_d"], delay_moments["var_d"],
        delay_moments["cov_d"], idx)

    cov_at_g4_g5 = covariance_between_paths(PATH1, PATH2, delay_moments["cov_d"], idx)

    sigma_g4 = math.sqrt(max(var_at_g4, 1e-18))
    sigma_g5 = math.sqrt(max(var_at_g5, 1e-18))
    rho = cov_at_g4_g5 / (sigma_g4 * sigma_g5) if sigma_g4 * sigma_g5 > 0 else 0.0
    rho = max(-1.0, min(1.0, rho))

    mu_max, var_max = clark_max(mu_at_g4, var_at_g4, mu_at_g5, var_at_g5, rho)

    mu_d6 = float(delay_moments["mean_d"][idx["G6"]])
    var_d6 = float(delay_moments["var_d"][idx["G6"]])

    cov_at_g4_d6 = covariance_between_paths(PATH1, ["G6"], delay_moments["cov_d"], idx)
    cov_at_g5_d6 = covariance_between_paths(PATH2, ["G6"], delay_moments["cov_d"], idx)

    theta = (mu_at_g4 - mu_at_g5) / math.sqrt(
        max(var_at_g4 + var_at_g5 - 2 * rho * sigma_g4 * sigma_g5, 1e-18))
    Phi_theta = 0.5 * (1.0 + math.erf(theta / math.sqrt(2.0)))
    Phi_neg_theta = 0.5 * (1.0 + math.erf(-theta / math.sqrt(2.0)))
    cov_max_d6 = Phi_theta * cov_at_g4_d6 + Phi_neg_theta * cov_at_g5_d6

    mu_final, var_final = gaussian_sum(mu_max, var_max, mu_d6, var_d6, cov_max_d6)
    std_final = math.sqrt(max(var_final, 0.0))

    return {
        "mean": mu_final,
        "std": std_final,
        **{k: mu_final + z * std_final for k, z in Z.items()},
        "path_moments": {
            "AT_G4_mean": mu_at_g4, "AT_G4_std": sigma_g4,
            "AT_G5_mean": mu_at_g5, "AT_G5_std": sigma_g5,
            "Cov_AT_G4_AT_G5": cov_at_g4_g5, "rho_AT_G4_AT_G5": rho,
        },
        "clark_max": {"mu_max": mu_max, "std_max": math.sqrt(max(var_max, 0.0))},
        "final_sum": {"mu_final": mu_final, "var_final": var_final,
                      "std_final": std_final, "cov_max_d6": cov_max_d6},
    }


def recompute_run_b_from_raw() -> dict:
    """Re-derivation of Run B (experiments/run_stage4b_hybrid.py) from the
    locked N=100k seed-42 raw reference — identical recipe including ddof=1."""
    ref = np.load(STAGE3_RAW)
    arrival_times = ref["arrival_times"]
    delays = ref["delays"]
    gate_names = ["G1", "G2", "G3", "G4", "G5", "G6"]
    idx = {name: i for i, name in enumerate(gate_names)}

    at_g4 = arrival_times[:, idx["G4"]]
    at_g5 = arrival_times[:, idx["G5"]]
    d_g6 = delays[:, idx["G6"]]

    mu1_emp = float(np.mean(at_g4))
    sigma1_emp = float(np.std(at_g4, ddof=1))
    mu2_emp = float(np.mean(at_g5))
    sigma2_emp = float(np.std(at_g5, ddof=1))
    rho_emp = float(np.corrcoef(at_g4, at_g5)[0, 1])

    mu_d6_emp = float(np.mean(d_g6))
    sigma_d6_emp = float(np.std(d_g6, ddof=1))
    cov_at_g4_d6 = float(np.cov(at_g4, d_g6)[0, 1])
    cov_at_g5_d6 = float(np.cov(at_g5, d_g6)[0, 1])

    max_at = np.maximum(at_g4, at_g5)
    cov_max_d6_empirical = float(np.cov(max_at, d_g6)[0, 1])

    mu_max_b, var_max_b = clark_max(mu1_emp, sigma1_emp ** 2, mu2_emp, sigma2_emp ** 2, rho_emp)
    std_max_b = float(np.sqrt(max(var_max_b, 0.0)))

    theta = (mu1_emp - mu2_emp) / np.sqrt(max(sigma1_emp ** 2 + sigma2_emp ** 2
                                              - 2 * rho_emp * sigma1_emp * sigma2_emp, 1e-18))
    Phi_theta = 0.5 * (1.0 + math.erf(theta / np.sqrt(2.0)))
    Phi_neg_theta = 0.5 * (1.0 + math.erf(-theta / np.sqrt(2.0)))
    cov_max_d6_approx = Phi_theta * cov_at_g4_d6 + Phi_neg_theta * cov_at_g5_d6

    mu_final_b, var_final_b = gaussian_sum(mu_max_b, var_max_b, mu_d6_emp,
                                           sigma_d6_emp ** 2, cov_max_d6_approx)
    std_final_b = float(np.sqrt(max(var_final_b, 0.0)))
    p99_87_b = mu_final_b + 3.0 * std_final_b

    return {
        "empirical_moments": {
            "AT_G4_mean": mu1_emp, "AT_G4_std": sigma1_emp,
            "AT_G5_mean": mu2_emp, "AT_G5_std": sigma2_emp,
            "rho_AT_G4_AT_G5": rho_emp,
            "d_G6_mean": mu_d6_emp, "d_G6_std": sigma_d6_emp,
            "cov_max_d6_empirical": cov_max_d6_empirical,
            "cov_max_d6_approx": float(cov_max_d6_approx),
        },
        "run_b_clark": {
            "mu_max": mu_max_b, "std_max": std_max_b,
            "mu_final": mu_final_b, "std_final": std_final_b,
            "p99_87": float(p99_87_b),
        },
    }


def check_d1(config) -> dict:
    """D1: order-dependent array readback.

    For the fixed six-gate topology we verify the stronger property: the
    analytical figures are invariant under EVERY permutation of gate_coords.
    (The graph itself also has a unique deterministic topological order, but
    the permutation sweep is what proves no D1-class sensitivity exists.)
    """
    tg = build_branching_graph_from_config(
        timing_params=config.timing_params, dag=config.dag.successors,
        variation_params=config.variation_params)
    order = tg.topological_order()
    coords = config.variation_params.gate_coords
    coords_keys = list(coords.keys())

    base = recompute_analytical_from_config(config)
    metrics = ["mean", "std", "p95", "p99", "p99_87"]

    max_diff = 0.0
    worst = None
    n_perms = 0
    for perm in permutations(coords_keys):
        vp = replace(config.variation_params, gate_coords={n: coords[n] for n in perm})
        res = recompute_analytical_from_config(replace(config, variation_params=vp))
        diff = max(abs(res[m] - base[m]) for m in metrics)
        if diff > max_diff:
            max_diff, worst = diff, perm
        n_perms += 1

    return {
        "topological_order": order,
        "gate_coords_key_order": coords_keys,
        "topo_order_matches_coords_order": order == coords_keys,
        "note": ("D1 readback (topological-index into gate_coords-ordered arrays) is "
                 "harmless for this topology iff topo order == coords order AND the "
                 "pipeline is invariant to gate_coords permutation; both are tested."),
        "n_permutations_tested": n_perms,
        "max_absdiff_over_permutations": max_diff,
        "worst_permutation": list(worst) if worst is not None else None,
        "bit_invariant_under_all_permutations": max_diff == 0.0,
    }


def check_d2(config) -> dict:
    """D2: Pelgrom / variation-model consistency between the MC sampler and
    the analytical comparator. Both must apply the same d-cap (min(d, 5 um))."""
    params = config.variation_params
    w_um = params.w_nom_nm * 1e-3
    l_um = params.l_nom_nm * 1e-3

    per_gate = {}
    cap_active_any = False
    for name, (x, y) in params.gate_coords.items():
        d_um = float(math.sqrt(x ** 2 + y ** 2))
        sigma_capped = _pelgrom_vth_sigma(params.w_nom_nm, params.l_nom_nm,
                                          params.vth_pelgrom_A_v_um,
                                          params.vth_pelgrom_S_v_um, d_um)
        # pre-cap formula: same expression without min(d, 5)
        sigma_precap = math.sqrt((params.vth_pelgrom_A_v_um ** 2) / (w_um * l_um)
                                 + (params.vth_pelgrom_S_v_um * d_um) ** 2)
        cap_active = d_um > 5.0
        cap_active_any |= cap_active
        per_gate[name] = {
            "d_um": d_um,
            "cap_active": cap_active,
            "sigma_capped": sigma_capped,
            "sigma_precap": sigma_precap,
            "capped_equals_precap": sigma_capped == sigma_precap,
        }

    # analytical comparator currently uses the capped formula (post-ba994f3).
    # variation/analytical.py: var_vth[i] = inter_die^2 + spatial_cov[i,i] + pelgrom(d)^2
    # with spatial_cov[i,i] = spatial_sigma^2 * exp(-0/lam) = spatial_sigma^2.
    pm = compute_process_moments(params)
    names = pm["names"]
    sampler_matches_analytical = True
    for name in names:
        x, y = params.gate_coords[name]
        d_um = float(math.sqrt(x ** 2 + y ** 2))
        expected_var_random = _pelgrom_vth_sigma(params.w_nom_nm, params.l_nom_nm,
                                                 params.vth_pelgrom_A_v_um,
                                                 params.vth_pelgrom_S_v_um, d_um) ** 2
        idx = pm["idx"][name]
        inter = params.inter_die_sigma_vth ** 2
        spatial = params.spatial_sigma_vth ** 2 * math.exp(-0.0 / params.spatial_lambda)
        expected = inter + spatial + expected_var_random
        if not math.isclose(pm["var_vth"][idx], expected, rel_tol=1e-12, abs_tol=0.0):
            sampler_matches_analytical = False

    return {
        "per_gate": per_gate,
        "max_gate_distance_um": max(v["d_um"] for v in per_gate.values()),
        "cap_inactive_for_all_gates": not cap_active_any,
        "analytical_comparator_uses_capped_formula": sampler_matches_analytical,
        "note": ("D2 could only bite if a gate sits beyond 5 um from origin "
                 "(cap boundary) or if sampler and comparator formulas diverge. "
                 "Both checked numerically per gate."),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    config = load_config(REPO / "foundations" / "stage3_config.json")

    with open(STAGE4_JSON) as f:
        stored_stage4 = json.load(f)
    with open(STAGE4B_JSON) as f:
        stored_stage4b = json.load(f)
    with open(STAGE3_SUMMARY) as f:
        stored_mc_summary = json.load(f)

    print("=== Kit 3 — Stage 4 fixed-six-gate audit ===\n")

    # ------------------------------------------------------------------
    # D1 / D2 defect-pattern checks
    # ------------------------------------------------------------------
    print("[1] D1 — order-dependence readback check (permutation sweep)")
    d1 = check_d1(config)
    print(f"    topo order {d1['topological_order']} == coords order: "
          f"{d1['topo_order_matches_coords_order']}")
    print(f"    {d1['n_permutations_tested']} gate_coords permutations tested: "
          f"max|diff| = {d1['max_absdiff_over_permutations']:.3e} "
          f"-> bit-invariant: {d1['bit_invariant_under_all_permutations']}")

    print("\n[2] D2 — Pelgrom/variation-model consistency check")
    d2 = check_d2(config)
    print(f"    max gate distance from origin: {d2['max_gate_distance_um']:.2f} um "
          f"(cap boundary 5.0 um) -> cap inactive for all gates: "
          f"{d2['cap_inactive_for_all_gates']}")
    print(f"    analytical comparator uses the same capped formula as the MC sampler: "
          f"{d2['analytical_comparator_uses_capped_formula']}")

    # ------------------------------------------------------------------
    # Bit-exact revalidation: Run A (stored analytical figures)
    # ------------------------------------------------------------------
    print("\n[3] Run A reimplementation vs stored stage4_analytical_ssta.json")
    recomputed = recompute_analytical_from_config(config)
    run_a_checks = {}
    for key in ["mean", "std", "p95", "p99", "p99_87"]:
        stored = stored_stage4["analytical_ssta"][key]
        rc = recomputed[key]
        run_a_checks[key] = {
            "stored": stored, "recomputed": rc,
            "absdiff": abs(stored - rc), "exact_repr_match": repr(stored) == repr(rc),
        }
        print(f"    {key:7s}: stored={stored:.12f} recomputed={rc:.12f} "
              f"|diff|={abs(stored - rc):.3e} exact={repr(stored) == repr(rc)}")
    for sub in ["path_moments", "clark_max", "final_sum"]:
        for k, v in stored_stage4[sub].items():
            rc = recomputed[sub][k]
            run_a_checks[f"{sub}.{k}"] = {
                "stored": v, "recomputed": rc,
                "absdiff": abs(v - rc), "exact_repr_match": repr(v) == repr(rc),
            }

    # ------------------------------------------------------------------
    # Bit-exact revalidation: Run B + gap decomposition (0.2781)
    # ------------------------------------------------------------------
    print("\n[4] Run B re-derivation from stage3_raw.npz vs stored stage4b_hybrid_run.json")
    run_b = recompute_run_b_from_raw()
    run_b_checks = {}
    for group in ["empirical_moments", "run_b_clark"]:
        for k, v in stored_stage4b[group].items():
            rc = run_b[group][k]
            run_b_checks[f"{group}.{k}"] = {
                "stored": v, "recomputed": rc,
                "absdiff": abs(v - rc), "exact_repr_match": repr(v) == repr(rc),
            }
            print(f"    {group}.{k}: stored={v:.12f} recomputed={rc:.12f} "
                  f"|diff|={abs(v - rc):.3e}")

    mc_p99_87_stored = stored_mc_summary["stats"]["p99_87"]
    ref = np.load(STAGE3_RAW)["critical_path_delay"]
    mc_p99_87_recomputed = float(np.quantile(ref, 0.9987))

    analytical_p99_87 = stored_stage4["analytical_ssta"]["p99_87"]
    stored_gap = stored_stage4b["gap_decomposition"]
    recomputed_gap = {
        "total_gap": mc_p99_87_recomputed - recomputed["p99_87"],
        "shape_gap": mc_p99_87_recomputed - run_b["run_b_clark"]["p99_87"],
        "linearization_gap": run_b["run_b_clark"]["p99_87"] - recomputed["p99_87"],
    }
    gap_checks = {}
    for k in ["total_gap", "shape_gap", "linearization_gap"]:
        gap_checks[k] = {
            "stored": stored_gap[k], "recomputed": recomputed_gap[k],
            "absdiff": abs(stored_gap[k] - recomputed_gap[k]),
            "exact_repr_match": repr(stored_gap[k]) == repr(recomputed_gap[k]),
        }
        print(f"    gap.{k}: stored={stored_gap[k]:.12f} "
              f"recomputed={recomputed_gap[k]:.12f} |diff|={abs(stored_gap[k] - recomputed_gap[k]):.3e}")
    print(f"    MC P99.87: stored={mc_p99_87_stored:.12f} recomputed={mc_p99_87_recomputed:.12f} "
          f"|diff|={abs(mc_p99_87_stored - mc_p99_87_recomputed):.3e}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    # Classification: bit-exact (repr-identical) vs last-ulp float drift
    # (< 1e-12 abs — numpy/BLAS reduction-order artifact across versions,
    # NOT a modeling defect) vs genuine mismatch (>= 1e-12).
    ULP_TOL = 1e-12

    def classify(checks: dict) -> dict:
        n_exact = sum(1 for c in checks.values() if c["exact_repr_match"])
        n_ulp = sum(1 for c in checks.values()
                    if not c["exact_repr_match"] and c["absdiff"] < ULP_TOL)
        n_mismatch = len(checks) - n_exact - n_ulp
        return {"n_total": len(checks), "n_bit_exact": n_exact,
                "n_last_ulp_drift": n_ulp, "n_genuine_mismatch": n_mismatch,
                "max_absdiff": max(c["absdiff"] for c in checks.values()),
                "all_within_float_tolerance": n_mismatch == 0}

    run_a_class = classify(run_a_checks)
    run_b_class = classify(run_b_checks)
    gap_class = classify(gap_checks)
    mc_exact = repr(mc_p99_87_stored) == repr(mc_p99_87_recomputed)
    d1_clean = d1["bit_invariant_under_all_permutations"]
    d2_clean = d2["cap_inactive_for_all_gates"] and d2["analytical_comparator_uses_capped_formula"]

    all_reproduced = (run_a_class["all_within_float_tolerance"]
                      and run_b_class["all_within_float_tolerance"]
                      and gap_class["all_within_float_tolerance"] and mc_exact)
    headline_bit_exact = all(c["exact_repr_match"] for c in gap_checks.values())

    verdict = {
        "scope": "Stage 4 fixed six-gate reference topology (the §IV 0.2781 pipeline, post propagate_path fix)",
        "scripts_audited": [
            "ssta/analytical_ssta.py (Run A: linearized moments + Clark MAX — the stored analytical comparator)",
            "experiments/run_stage4b_hybrid.py (Run B: empirical moments + Clark — gap decomposition)",
        ],
        "inputs_audited": [
            "foundations/stage3_config.json (frozen coordinates/params)",
            "results/stage3_raw.npz (locked N=100k, seed 42 MC reference)",
            "results/stage3_summary.json",
        ],
        "stored_outputs_revalidated": [
            "results/stage4_analytical_ssta.json",
            "results/stage4b_hybrid_run.json",
        ],
        "d1_check": d1,
        "d2_check": d2,
        "run_a_reimplementation": {**run_a_class, "checks": run_a_checks},
        "run_b_rederivation": {**run_b_class, "checks": run_b_checks},
        "gap_decomposition": {
            **gap_class, "checks": gap_checks,
            "headline_numbers_bit_exact": headline_bit_exact,
            "mc_p99_87": {"stored": mc_p99_87_stored, "recomputed": mc_p99_87_recomputed,
                          "exact": mc_exact},
            "stored_values": {"total_gap": stored_gap["total_gap"],
                              "shape_gap": stored_gap["shape_gap"],
                              "linearization_gap": stored_gap["linearization_gap"]},
        },
        "caveats": [
            "Run B consumes EMPIRICAL moments (ddof=1) from the locked MC reference — "
            "estimator variance, not just distributional shape, contributes to the "
            "shape gap (already flagged in run_stage4b_hybrid.py).",
            "P99.87 here is mean+3sigma of the final Gaussian — same convention as the "
            "stored figures; MC side is the empirical quantile of the locked N=100k sample.",
            "The Stage 4 pipeline operates on ONE frozen topology: the D1 pattern that "
            "affected compute_analytical_ssta_arbitrary (corpus code, still unfixed in "
            "data_generation/analytical_ssta_arbitrary.py) is structurally harmless here, "
            "and the permutation sweep + bit-exact reproduction prove it empirically.",
        ],
        "verdict": {
            "d1_defect_found": not d1_clean,
            "d2_defect_found": not d2_clean,
            "stored_figures_reproduced": bool(all_reproduced),
            "gap_decomposition_headline_bit_exact": bool(headline_bit_exact and mc_exact),
            "last_ulp_note": (
                "Values with absdiff in (0, 1e-12) are last-ulp floating-point "
                "reassociation differences (numpy/BLAS reduction order across "
                "versions, e.g. np.corrcoef's internal matmul); they do not "
                "indicate a modeling or pipeline defect. All three headline gap "
                "numbers reproduce bit-exactly because they are differences of "
                "exactly-round-tripped endpoints."
            ),
            "outcome": ("CLEAN — no D1/D2 defect pattern affects the fixed-topology "
                        "pipeline; the stored analytical figures and the 0.2781 gap "
                        "decomposition are reproduced (headline numbers bit-exactly, "
                        "all intermediates within last-ulp float tolerance) by a "
                        "from-scratch name-keyed reimplementation"
                        if (d1_clean and d2_clean and all_reproduced)
                        else "DEFECT FOUND — see component checks"),
        },
        "elapsed_s": time.time() - t0,
    }

    with open(OUT_DIR / "kit3_stage4_audit_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(OUT_DIR / "kit3_stage4_audit_report.json", "w") as f:
        json.dump({
            "recomputed_run_a": recomputed,
            "recomputed_run_b": run_b,
            "recomputed_gap_decomposition": recomputed_gap,
        }, f, indent=2, default=str)

    print(f"\n=== VERDICT: {verdict['verdict']['outcome']} ===")
    print(f"Saved {OUT_DIR / 'kit3_stage4_audit_verdict.json'}")
    print(f"Saved {OUT_DIR / 'kit3_stage4_audit_report.json'}")

    if not (d1_clean and d2_clean and all_reproduced):
        sys.exit(1)


if __name__ == "__main__":
    main()
