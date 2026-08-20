"""
Stage 4b — Hybrid Run: Empirical Moments + Clark MAX

Extracts AT_G4, AT_G5, d_G6 from stage3_raw.npz, computes empirical moments,
runs Clark's formula, and decomposes the gap between MC and analytical SSTA.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from timing.clark_max import clark_max
from foundations.config_loader import load_config
from ssta.statistical_sum import gaussian_sum


def main() -> None:
    t0 = time.time()
    config = load_config("foundations/stage3_config.json")

    ref = np.load("results/stage3_raw.npz")
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
    cov_max_d6_emp = float(np.cov(max_at, d_g6)[0, 1])

    # NOTE: clark_max and gaussian_sum treat inputs as population moments, but
    # the empirical moments above are sample estimates (ddof=1). This means the
    # "shape gap" partly reflects estimator variance rather than true distributional
    # shape differences. A rigorous fix would use ddof=0 or add a bias correction.

    mu_max_b, var_max_b = clark_max(mu1_emp, sigma1_emp ** 2, mu2_emp, sigma2_emp ** 2, rho_emp)
    std_max_b = np.sqrt(max(var_max_b, 0.0))

    theta = (mu1_emp - mu2_emp) / np.sqrt(max(sigma1_emp ** 2 + sigma2_emp ** 2 - 2 * rho_emp * sigma1_emp * sigma2_emp, 1e-18))
    Phi_theta = 0.5 * (1.0 + math.erf(theta / np.sqrt(2.0)))
    Phi_neg_theta = 0.5 * (1.0 + math.erf(-theta / np.sqrt(2.0)))
    cov_max_d6_approx = Phi_theta * cov_at_g4_d6 + Phi_neg_theta * cov_at_g5_d6

    mu_final_b, var_final_b = gaussian_sum(mu_max_b, var_max_b, mu_d6_emp, sigma_d6_emp ** 2, cov_max_d6_approx)
    std_final_b = np.sqrt(max(var_final_b, 0.0))
    p99_87_b = mu_final_b + 3.0 * std_final_b

    with open("results/stage4_analytical_ssta.json", "r") as f:
        analytical_report = json.load(f)
    analytical_p99_87 = analytical_report["analytical_ssta"]["p99_87"]

    with open("results/stage3_summary.json", "r") as f:
        mc_summary = json.load(f)
    mc_p99_87 = mc_summary["stats"]["p99_87"]

    total_gap = mc_p99_87 - analytical_p99_87
    shape_gap = mc_p99_87 - p99_87_b
    linearization_gap = p99_87_b - analytical_p99_87

    runtime_ms = (time.time() - t0) * 1000.0

    report = {
        "empirical_moments": {
            "AT_G4_mean": mu1_emp,
            "AT_G4_std": sigma1_emp,
            "AT_G5_mean": mu2_emp,
            "AT_G5_std": sigma2_emp,
            "rho_AT_G4_AT_G5": rho_emp,
            "d_G6_mean": mu_d6_emp,
            "d_G6_std": sigma_d6_emp,
            "cov_max_d6_empirical": cov_max_d6_emp,
            "cov_max_d6_approx": cov_max_d6_approx,
        },
        "run_b_clark": {
            "mu_max": mu_max_b,
            "std_max": float(std_max_b),
            "mu_final": mu_final_b,
            "std_final": float(std_final_b),
            "p99_87": float(p99_87_b),
        },
        "gap_decomposition": {
            "total_gap": float(total_gap),
            "shape_gap": float(shape_gap),
            "linearization_gap": float(linearization_gap),
            "check_sum": float(shape_gap + linearization_gap),
        },
        "runtime_ms": runtime_ms,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/stage4b_hybrid_run.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== Stage 4b — Hybrid Run (Empirical Moments + Clark) ===")
    print(f"AT_G4  : N({mu1_emp:.6f}, {sigma1_emp:.6f})")
    print(f"AT_G5  : N({mu2_emp:.6f}, {sigma2_emp:.6f})")
    print(f"rho    : {rho_emp:.6f}")
    print(f"d_G6   : N({mu_d6_emp:.6f}, {sigma_d6_emp:.6f})")
    print(f"MAX    : N({mu_max_b:.6f}, {std_max_b:.6f})")
    print(f"Final  : N({mu_final_b:.6f}, {std_final_b:.6f})")
    print(f"P99.87 : {p99_87_b:.6f}")
    print()
    print("=== Gap Decomposition ===")
    print(f"Total gap (MC - Analytical)    : {total_gap:+.6f}")
    print(f"Shape gap (MC - Run B)         : {shape_gap:+.6f}")
    print(f"Linearization gap (Run B - Ana): {linearization_gap:+.6f}")
    print(f"Check (shape + linearization)  : {shape_gap + linearization_gap:+.6f}")
    print()
    print("=== Three-Way Table (P99.87) ===")
    print(f"{'Quantity':<35} {'P99.87':>10} {'Gap vs MC':>12}")
    print("-" * 59)
    print(f"{'Monte Carlo (truth)':<35} {mc_p99_87:>10.6f} {'—':>12}")
    print(f"{'Run B (empirical moments + Clark)':<35} {p99_87_b:>10.6f} {shape_gap:>+12.6f}")
    print(f"{'Run A / Stage 4 (linearized + Clark)':<35} {analytical_p99_87:>10.6f} {total_gap:>+12.6f}")
    print(f"\nSaved results/stage4b_hybrid_run.json")


if __name__ == "__main__":
    main()
