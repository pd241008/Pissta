"""
Stage 5 — Tail-Aware SSTA with Skew-Normal MAX Approximation

Reuses Stage 4 linearized delay moments but replaces Clark's Gaussian MAX
with a skew-normal 3-moment match fitted to empirical max(AT_G4, AT_G5) skewness.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from foundations.config_loader import load_config
from ssta.statistical_sum import covariance_between_paths, gaussian_sum, propagate_path
from timing.clark_max import clark_max
from timing.delay import compute_delay_moments, nominal_delay
from timing.tail_aware_max import sample_skew_normal, tail_aware_max
from variation.analytical import compute_process_moments


def load_stage3_reference(path: str = "results/stage3_raw.npz") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage 3 reference not found at {path}. Run run_stage3.py first.")
    return np.load(path)


def load_stage3_summary(path: str = "results/stage3_summary.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_stage4_results(path: str = "results/stage4_analytical_ssta.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_stage4b_results(path: str = "results/stage4b_hybrid_run.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_stage3_reproducibility(path: str = "results/stage3_reproducibility_N100k.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_linearized_path_moments(config) -> dict:
    timing_params = config.timing_params
    variation_params = config.variation_params

    process_moments = compute_process_moments(variation_params)
    delay_moments = compute_delay_moments(timing_params, variation_params, process_moments)

    path1 = ["G1", "G2", "G4"]
    path2 = ["G1", "G3", "G5"]

    mu_at_g4, var_at_g4 = propagate_path(
        path1,
        delay_moments["mean_d"],
        delay_moments["var_d"],
        delay_moments["cov_d"],
        delay_moments["idx"],
    )
    mu_at_g5, var_at_g5 = propagate_path(
        path2,
        delay_moments["mean_d"],
        delay_moments["var_d"],
        delay_moments["cov_d"],
        delay_moments["idx"],
    )

    cov_at_g4_g5 = covariance_between_paths(
        path1, path2, delay_moments["cov_d"], delay_moments["idx"]
    )

    sigma_g4 = math.sqrt(max(var_at_g4, 1e-18))
    sigma_g5 = math.sqrt(max(var_at_g5, 1e-18))
    rho = cov_at_g4_g5 / (sigma_g4 * sigma_g5) if sigma_g4 * sigma_g5 > 0 else 0.0
    rho = max(-1.0, min(1.0, rho))

    mu_d6 = float(delay_moments["mean_d"][delay_moments["idx"]["G6"]])
    var_d6 = float(delay_moments["var_d"][delay_moments["idx"]["G6"]])

    cov_at_g4_d6 = covariance_between_paths(
        path1, ["G6"], delay_moments["cov_d"], delay_moments["idx"]
    )
    cov_at_g5_d6 = covariance_between_paths(
        path2, ["G6"], delay_moments["cov_d"], delay_moments["idx"]
    )

    return {
        "mu_at_g4": mu_at_g4,
        "var_at_g4": var_at_g4,
        "mu_at_g5": mu_at_g5,
        "var_at_g5": var_at_g5,
        "cov_at_g4_g5": cov_at_g4_g5,
        "rho": rho,
        "mu_d6": mu_d6,
        "var_d6": var_d6,
        "cov_at_g4_d6": cov_at_g4_d6,
        "cov_at_g5_d6": cov_at_g5_d6,
    }


def extract_empirical_max_skewness(stage3_data: dict, n_samples: int = 100_000) -> float:
    at_g4 = stage3_data["arrival_times"][:, 3]  # G4 index
    at_g5 = stage3_data["arrival_times"][:, 4]  # G5 index
    max_at = np.maximum(at_g4, at_g5)
    m = np.mean(max_at)
    s = np.std(max_at, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean((max_at - m) ** 3) / (s ** 3))


def main() -> None:
    t0 = time.time()
    config = load_config("foundations/stage3_config.json")
    path_moments = compute_linearized_path_moments(config)

    stage3_data = load_stage3_reference()
    stage4_results = load_stage4_results()
    stage4b_results = load_stage4b_results()
    mc_summary = load_stage3_summary()

    empirical_skewness = extract_empirical_max_skewness(stage3_data)
    print(f"Empirical skewness of max(AT_G4, AT_G5): {empirical_skewness:.6f}")

    at_g4 = stage3_data["arrival_times"][:, 3]
    at_g5 = stage3_data["arrival_times"][:, 4]
    max_at = np.maximum(at_g4, at_g5)
    mu_max_emp = float(np.mean(max_at))
    var_max_emp = float(np.var(max_at, ddof=1))

    clark_result = tail_aware_max(
        path_moments["mu_at_g4"],
        path_moments["var_at_g4"],
        path_moments["mu_at_g5"],
        path_moments["var_at_g5"],
        path_moments["rho"],
        empirical_skewness=None,
    )

    tail_aware_result = tail_aware_max(
        mu_max_emp,
        var_max_emp,
        mu_max_emp,
        var_max_emp,
        stage4b_results["empirical_moments"]["rho_AT_G4_AT_G5"],
        empirical_skewness=empirical_skewness,
    )

    n_approx_samples = 100_000
    rng = np.random.default_rng(42)

    if tail_aware_result["method"] == "skew_normal_3moment":
        xi = tail_aware_result["xi"]
        omega = tail_aware_result["omega"]
        alpha = tail_aware_result["alpha"]
        max_samples = sample_skew_normal(n_approx_samples, xi, omega, alpha, rng)
    else:
        max_samples = rng.normal(
            tail_aware_result["mu_max"],
            tail_aware_result["std_max"],
            n_approx_samples,
        )

    mu_d6 = path_moments["mu_d6"]
    var_d6 = path_moments["var_d6"]
    sigma_d6 = math.sqrt(var_d6)
    theta = (path_moments["mu_at_g4"] - path_moments["mu_at_g5"]) / math.sqrt(max(path_moments["var_at_g4"] + path_moments["var_at_g5"] - 2 * path_moments["rho"] * math.sqrt(path_moments["var_at_g4"]) * math.sqrt(path_moments["var_at_g5"]), 1e-18))
    Phi_theta = 0.5 * (1.0 + math.erf(theta / math.sqrt(2.0)))
    Phi_neg_theta = 0.5 * (1.0 + math.erf(-theta / math.sqrt(2.0)))
    cov_max_d6_linearized = Phi_theta * path_moments["cov_at_g4_d6"] + Phi_neg_theta * path_moments["cov_at_g5_d6"]

    cov_max_d6 = stage4b_results["empirical_moments"]["cov_max_d6_empirical"]
    beta = cov_max_d6 / tail_aware_result["std_max"] ** 2 if tail_aware_result["std_max"] > 1e-18 else 0.0
    residual_var = max(var_d6 - beta ** 2 * tail_aware_result["std_max"] ** 2, 1e-18)
    residual_std = math.sqrt(residual_var)

    d6_samples = mu_d6 + beta * (max_samples - tail_aware_result["mu_max"]) + rng.normal(0.0, residual_std, n_approx_samples)
    final_samples = max_samples + d6_samples

    tail_aware_p95 = float(np.quantile(final_samples, 0.95))
    tail_aware_p99 = float(np.quantile(final_samples, 0.99))
    tail_aware_p99_87 = float(np.quantile(final_samples, 0.9987))
    tail_aware_mean = float(np.mean(final_samples))
    tail_aware_std = float(np.std(final_samples, ddof=1))

    mc_p99_87 = mc_summary["stats"]["p99_87"]
    stage4_p99_87 = stage4_results["analytical_ssta"]["p99_87"]
    stage4b_p99_87 = stage4b_results["run_b_clark"]["p99_87"]

    total_gap = mc_p99_87 - stage4_p99_87
    shape_gap = mc_p99_87 - stage4b_p99_87
    tail_aware_gap = mc_p99_87 - tail_aware_p99_87
    gap_closure = shape_gap - tail_aware_gap
    closure_fraction = gap_closure / shape_gap if shape_gap != 0 else 0.0

    repro = load_stage3_reproducibility()
    mc_p99_87_values = [run["branch_p99_87"] for run in repro["runs"]]
    mc_p99_87_pooled = float(np.mean(mc_p99_87_values))
    tail_aware_gap_pooled = mc_p99_87_pooled - tail_aware_p99_87
    gap_closure_pooled = shape_gap - tail_aware_gap_pooled
    closure_fraction_pooled = gap_closure_pooled / shape_gap if shape_gap != 0 else 0.0

    absolute_gaps = [mc_val - tail_aware_p99_87 for mc_val in mc_p99_87_values]
    absolute_gap_mean = float(np.mean(absolute_gaps))
    absolute_gap_std = float(np.std(absolute_gaps, ddof=1)) if len(absolute_gaps) > 1 else 0.0

    multi_seed = []
    for run in repro["runs"]:
        seed = run["seed"]
        mc_p99_87_seed = run["branch_p99_87"]
        abs_gap = mc_p99_87_seed - tail_aware_p99_87
        multi_seed.append({
            "seed": seed,
            "mc_p99_87": mc_p99_87_seed,
            "tail_aware_p99_87": tail_aware_p99_87,
            "absolute_gap": abs_gap,
        })

    runtime_ms = (time.time() - t0) * 1000.0

    report = {
        "config_file": "foundations/stage3_config.json",
        "empirical_skewness_of_max": empirical_skewness,
        "tail_aware_max_params": {
            "xi": tail_aware_result["xi"],
            "omega": tail_aware_result["omega"],
            "alpha": tail_aware_result["alpha"],
            "method": tail_aware_result["method"],
        },
        "tail_aware_ssta": {
            "mean": tail_aware_mean,
            "std": tail_aware_std,
            "p95": tail_aware_p95,
            "p99": tail_aware_p99,
            "p99_87": tail_aware_p99_87,
        },
        "comparison": {
            "monte_carlo": {
                "mean": mc_summary["stats"]["mean"],
                "std": mc_summary["stats"]["std"],
                "p95": mc_summary["stats"]["p95"],
                "p99": mc_summary["stats"]["p99"],
                "p99_87": mc_p99_87,
            },
            "clark_linearized": {
                "mean": stage4_results["analytical_ssta"]["mean"],
                "std": stage4_results["analytical_ssta"]["std"],
                "p95": stage4_results["analytical_ssta"]["p95"],
                "p99": stage4_results["analytical_ssta"]["p99"],
                "p99_87": stage4_p99_87,
            },
            "clark_empirical": {
                "mean": stage4b_results["run_b_clark"]["mu_final"],
                "std": stage4b_results["run_b_clark"]["std_final"],
                "p95": float(np.quantile(stage3_data["critical_path_delay"], 0.95)),
                "p99": float(np.quantile(stage3_data["critical_path_delay"], 0.99)),
                "p99_87": stage4b_p99_87,
            },
            "tail_aware_linearized": {
                "mean": tail_aware_mean,
                "std": tail_aware_std,
                "p95": tail_aware_p95,
                "p99": tail_aware_p99,
                "p99_87": tail_aware_p99_87,
            },
        },
        "gap_analysis": {
            "total_gap_mc_vs_clark_lin": total_gap,
            "shape_gap_mc_vs_clark_emp": shape_gap,
            "tail_aware_gap_vs_seed42": tail_aware_gap,
            "tail_aware_gap_vs_pooled_mc": tail_aware_gap_pooled,
            "gap_closure_vs_shape_seed42": gap_closure,
            "closure_fraction_seed42": closure_fraction,
            "gap_closure_vs_shape_pooled": gap_closure_pooled,
            "closure_fraction_pooled": closure_fraction_pooled,
            "absolute_gaps_per_seed": {str(m["seed"]): m["absolute_gap"] for m in multi_seed},
            "absolute_gap_mean": absolute_gap_mean,
            "absolute_gap_std": absolute_gap_std,
            "pooled_mc_p99_87": mc_p99_87_pooled,
        },
        "runtime_ms": runtime_ms,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/stage5_tail_aware_ssta.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Stage 5 Tail-Aware SSTA ===")
    print(f"Runtime (tail-aware) : {runtime_ms:.3f} ms")
    print(f"Empirical max skew   : {empirical_skewness:.6f}")
    print(f"Skew-normal params   : xi={tail_aware_result['xi']:.6f}, omega={tail_aware_result['omega']:.6f}, alpha={tail_aware_result['alpha']:.6f}")
    print()
    print(f"{'Method':<30} {'Mean':>10} {'Std':>10} {'P95':>10} {'P99':>10} {'P99.87':>10}")
    print("-" * 80)
    for method, stats in report["comparison"].items():
        print(f"{method:<30} {stats['mean']:>10.6f} {stats['std']:>10.6f} {stats['p95']:>10.6f} {stats['p99']:>10.6f} {stats['p99_87']:>10.6f}")
    print()
    print("=== Gap Analysis (Absolute Gaps) ===")
    print(f"Total gap (MC - Clark/lin)    : {total_gap:+.6f}")
    print(f"Shape gap (MC - Clark/emp)    : {shape_gap:+.6f}")
    print(f"Tail-aware gap (vs seed 42)   : {tail_aware_gap:+.6f}")
    print(f"Tail-aware gap (vs pooled MC) : {tail_aware_gap_pooled:+.6f}")
    print(f"Absolute gap mean (3 seeds)   : {absolute_gap_mean:+.6f}")
    print(f"Absolute gap std  (3 seeds)   : {absolute_gap_std:+.6f}")
    print()
    print("=== Closure Analysis ===")
    print(f"Closure vs shape (seed 42)    : {gap_closure:+.6f}  ({closure_fraction:.2%})")
    print(f"Closure vs shape (pooled MC)  : {gap_closure_pooled:+.6f}  ({closure_fraction_pooled:.2%})")
    print(f"Pooled MC P99.87              : {mc_p99_87_pooled:.6f}")
    if closure_fraction_pooled < 0.40:
        print("WARNING: Closure < 40%. Check skew-normal fit or skewness computation.")
    elif closure_fraction_pooled > 1.20:
        print("WARNING: Closure > 120%. Skew correction may be overfitting.")
    else:
        print("PASS: Closure in expected 40-120% range.")
    print()
    print("=== Multi-Seed Validation (Absolute Gaps) ===")
    for entry in multi_seed:
        seed = entry["seed"]
        print(f"  Seed {seed}: MC={entry['mc_p99_87']:.6f}, Tail={entry['tail_aware_p99_87']:.6f}, abs_gap={entry['absolute_gap']:+.6f}")
    print(f"\nSaved results/stage5_tail_aware_ssta.json")


if __name__ == "__main__":
    main()
