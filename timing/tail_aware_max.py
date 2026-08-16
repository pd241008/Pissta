"""
Tail-aware MAX approximation using skew-normal 3-moment matching.

Replaces Clark's Gaussian moment-matched MAX with a skew-normal fit that
captures the right-skew of max(AT_G4, AT_G5).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from timing.clark_max import clark_max


def _phi(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skewness_to_delta(gamma: float) -> float:
    """Solve for delta in the skew-normal skewness equation numerically."""
    if abs(gamma) < 1e-12:
        return 0.0

    def skewness_eq(delta: float) -> float:
        if delta <= -1.0 or delta >= 1.0:
            return float("inf")
        delta2 = delta * delta
        var = 1.0 - 2.0 * delta2 / math.pi
        if var <= 0:
            return float("inf")
        mean_term = delta * math.sqrt(2.0 / math.pi)
        skew = ((4.0 - math.pi) / 2.0) * (mean_term ** 3) / (var ** 1.5)
        return skew - gamma

    lo, hi = -0.9999, 0.9999
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = skewness_eq(mid)
        if abs(val) < 1e-12:
            return mid
        if skewness_eq(lo) * val <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def skew_normal_from_moments(mean: float, variance: float, skewness: float) -> Tuple[float, float, float]:
    """Fit SN(xi, omega, alpha) to given mean, variance, skewness."""
    if variance <= 0:
        raise ValueError("Variance must be positive.")
    if abs(skewness) < 1e-12:
        return mean, math.sqrt(variance), 0.0

    delta = _skewness_to_delta(skewness)
    delta2 = delta * delta
    var_factor = 1.0 - 2.0 * delta2 / math.pi
    if var_factor <= 0:
        raise ValueError("Invalid skewness for skew-normal fit.")

    omega = math.sqrt(variance / var_factor)
    xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
    alpha = delta / math.sqrt(1.0 - delta2) if abs(delta) < 1.0 else float("inf")

    return xi, omega, alpha


def skew_normal_pdf(x: float, xi: float, omega: float, alpha: float) -> float:
    z = (x - xi) / omega
    return (2.0 / omega) * _phi(z) * _Phi(alpha * z)


def skew_normal_cdf(x: float, xi: float, omega: float, alpha: float) -> float:
    z = (x - xi) / omega
    return _Phi(z) - 2.0 * _Phi(-alpha * z) * _Phi(z)


def skew_normal_ppf(p: float, xi: float, omega: float, alpha: float) -> float:
    lo, hi = xi - 10.0 * omega, xi + 10.0 * omega
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if skew_normal_cdf(mid, xi, omega, alpha) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sample_skew_normal(n: int, xi: float, omega: float, alpha: float, rng) -> np.ndarray:
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    return xi + omega * (delta * np.abs(u) + math.sqrt(1.0 - delta * delta) * v)


def tail_aware_max(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
    rho: float,
    empirical_skewness: float | None = None,
) -> dict:
    """Compute tail-aware MAX approximation.

    If empirical_skewness is provided, fits a skew-normal to the max's
    first three moments. Otherwise falls back to Clark's Gaussian match.
    """
    sigma1 = math.sqrt(max(var1, 1e-18))
    sigma2 = math.sqrt(max(var2, 1e-18))

    mu_max, var_max = clark_max(mu1, var1, mu2, var2, rho)
    std_max = math.sqrt(max(var_max, 0.0))

    if empirical_skewness is None:
        return {
            "method": "clark_gaussian",
            "mu_max": mu_max,
            "std_max": std_max,
            "xi": mu_max,
            "omega": std_max,
            "alpha": 0.0,
            "skewness": 0.0,
        }

    xi, omega, alpha = skew_normal_from_moments(mu_max, var_max, empirical_skewness)

    return {
        "method": "skew_normal_3moment",
        "mu_max": mu_max,
        "std_max": std_max,
        "xi": xi,
        "omega": omega,
        "alpha": alpha,
        "skewness": empirical_skewness,
    }
