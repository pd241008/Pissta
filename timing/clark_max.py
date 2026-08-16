"""
Clark's analytical MAX approximation for two correlated Gaussians.

Given X ~ N(μ1, σ1²) and Y ~ N(μ2, σ2²) with correlation ρ,
approximates max(X,Y) as a Gaussian with moments:

  a = sqrt(σ1² + σ2² - 2ρσ1σ2)
  θ = (μ1 - μ2) / a
  μ_max = μ1 Φ(θ) + μ2 Φ(-θ) + a φ(θ)
  σ²_max = (μ1²+σ1²)Φ(θ) + (μ2²+σ2²)Φ(-θ) + (μ1+μ2)a φ(θ) - μ_max²

Φ = standard normal CDF, φ = standard normal PDF.
"""

from __future__ import annotations

import math
from typing import Tuple


def _phi(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clark_max(
    mu1: float,
    var1: float,
    mu2: float,
    var2: float,
    rho: float,
) -> Tuple[float, float]:
    sigma1 = math.sqrt(max(var1, 0.0))
    sigma2 = math.sqrt(max(var2, 0.0))

    a = math.sqrt(max(var1 + var2 - 2.0 * rho * sigma1 * sigma2, 1e-18))
    theta = (mu1 - mu2) / a

    Phi_theta = _Phi(theta)
    Phi_neg_theta = _Phi(-theta)
    phi_theta = _phi(theta)

    mu_max = mu1 * Phi_theta + mu2 * Phi_neg_theta + a * phi_theta
    var_max = (
        (mu1 * mu1 + var1) * Phi_theta
        + (mu2 * mu2 + var2) * Phi_neg_theta
        + (mu1 + mu2) * a * phi_theta
        - mu_max * mu_max
    )

    return float(mu_max), float(max(var_max, 0.0))
