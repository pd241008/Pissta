
"""
Physics-Informed Surrogate SSTA
Stage 1: Monte Carlo Ground-Truth Timing Baseline

Purpose
-------
Build a small, transparent Monte Carlo timing engine that will later become
the reference implementation for:
    1. correlated process variation,
    2. analytical SSTA,
    3. tail-aware MAX,
    4. GNN surrogate modeling,
    5. uncertainty calibration.

This first version intentionally uses simple Gaussian process variation and
an Alpha-Power-style gate-delay equation. Numerical values below are DEMO
parameters for software validation only; they are not tied to a fabrication
technology or claimed silicon measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ProcessParams:
    """Nominal process parameters and 1-sigma variations."""
    l_nom_nm: float = 45.0
    l_sigma_nm: float = 2.0

    w_nom_nm: float = 90.0
    w_sigma_nm: float = 3.0

    vth_nom_v: float = 0.40
    vth_sigma_v: float = 0.02


@dataclass(frozen=True)
class GateParams:
    """Simple gate parameters for the demonstration model."""
    k: float = 1.0
    alpha: float = 1.3
    vdd_v: float = 1.0


@dataclass(frozen=True)
class Gate:
    """One gate in a simple feed-forward timing chain."""
    name: str
    load_ff: float


def sample_process(
    n_samples: int,
    process: ProcessParams,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Draw independent Gaussian process samples.

    Returns
    -------
    dict
        Arrays of shape (n_samples,).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    l = rng.normal(process.l_nom_nm, process.l_sigma_nm, n_samples)
    w = rng.normal(process.w_nom_nm, process.w_sigma_nm, n_samples)
    vth = rng.normal(process.vth_nom_v, process.vth_sigma_v, n_samples)

    # Reject physically nonsensical samples for this demonstration.
    if np.any(l <= 0) or np.any(w <= 0):
        raise RuntimeError("Generated non-positive geometry. Adjust parameters.")

    return {"L_nm": l, "W_nm": w, "Vth_v": vth}


def alpha_power_delay(
    vth_v: np.ndarray,
    load_ff: float,
    gate_params: GateParams,
) -> np.ndarray:
    r"""
    Alpha-Power-style delay relation:

        d ~ C_load * Vdd / [ k * (Vdd - Vth)^alpha ]

    The load is treated as a normalized capacitance proxy in this first
    software-validation stage.
    """
    vdd = gate_params.vdd_v
    denominator = gate_params.k * np.power(
        np.maximum(vdd - vth_v, 1e-6),
        gate_params.alpha,
    )

    delay = load_ff * vdd / denominator
    return delay


def apply_geometry_sensitivity(
    base_delay: np.ndarray,
    l_nm: np.ndarray,
    w_nm: np.ndarray,
    process: ProcessParams,
) -> np.ndarray:
    """
    Add simple normalized first-order geometry sensitivity.

    This is deliberately a software-test approximation, not a calibrated
    transistor model. It allows L and W variation to influence delay before
    we replace this block with a technology-calibrated physical model.
    """
    l_ratio = l_nm / process.l_nom_nm
    w_ratio = w_nm / process.w_nom_nm

    # Longer channel -> slower; wider device -> faster.
    geometry_factor = l_ratio / np.sqrt(np.maximum(w_ratio, 1e-6))
    return base_delay * geometry_factor


def run_chain_monte_carlo(
    n_samples: int = 10_000,
    seed: int = 42,
    process: ProcessParams | None = None,
    gate_params: GateParams | None = None,
    gates: Sequence[Gate] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo timing for a feed-forward gate chain.

    Because the chain has no reconvergent branches, the critical-path delay
    is simply the sum of the gate delays in each sample.
    """
    process = process or ProcessParams()
    gate_params = gate_params or GateParams()

    if gates is None:
        gates = (
            Gate("INV1", 1.0),
            Gate("INV2", 1.2),
            Gate("INV3", 1.4),
            Gate("INV4", 1.6),
            Gate("INV5", 1.8),
        )

    rng = np.random.default_rng(seed)
    samples = sample_process(n_samples, process, rng)

    total_delay = np.zeros(n_samples, dtype=float)
    per_gate = {}

    for gate in gates:
        delay = alpha_power_delay(
            samples["Vth_v"],
            gate.load_ff,
            gate_params,
        )
        delay = apply_geometry_sensitivity(
            delay,
            samples["L_nm"],
            samples["W_nm"],
            process,
        )

        per_gate[gate.name] = delay
        total_delay += delay

    return {
        "L_nm": samples["L_nm"],
        "W_nm": samples["W_nm"],
        "Vth_v": samples["Vth_v"],
        "total_delay": total_delay,
        **{f"delay_{name}": value for name, value in per_gate.items()},
    }


def summarize_delay(delay: np.ndarray) -> Dict[str, float]:
    """Return the key timing statistics used throughout the project."""
    if delay.ndim != 1 or delay.size == 0:
        raise ValueError("delay must be a non-empty 1-D array.")

    return {
        "mean": float(np.mean(delay)),
        "std": float(np.std(delay, ddof=1)),
        "p95": float(np.quantile(delay, 0.95)),
        "p99": float(np.quantile(delay, 0.99)),
        "p99_87": float(np.quantile(delay, 0.9987)),
        "min": float(np.min(delay)),
        "max": float(np.max(delay)),
    }


def main() -> None:
    results = run_chain_monte_carlo(n_samples=10_000, seed=42)
    stats = summarize_delay(results["total_delay"])

    print("=== Monte Carlo Ground-Truth Baseline ===")
    print(f"Samples : 10,000")
    print(f"Mean    : {stats['mean']:.6f}")
    print(f"Std     : {stats['std']:.6f}")
    print(f"P95     : {stats['p95']:.6f}")
    print(f"P99     : {stats['p99']:.6f}")
    print(f"P99.87  : {stats['p99_87']:.6f}")
    print(f"Min     : {stats['min']:.6f}")
    print(f"Max     : {stats['max']:.6f}")


if __name__ == "__main__":
    main()
