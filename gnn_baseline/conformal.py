"""
Stage 7 — Split conformal calibration for the DAG-GNN surrogate.

Wraps a trained mean/std predictor with a distribution-free conformal
procedure. The predictor outputs, per graph, a predicted critical-path delay
mean (mean_pred) and standard deviation (std_pred). Conformal prediction turns
those point/scale estimates into prediction intervals with a *guaranteed*
(under exchangeability) marginal coverage rate, e.g. "90% of true delays fall
in the interval", WITHOUT assuming the residual distribution is Gaussian.

Why the studentized residual:
    score = |y_true - mean_pred| / std_pred
The model's own predicted std acts as a heteroskedasticity-aware scale, so the
conformal interval is narrow where the model is confident and wide where it is
not — more informative than a fixed-width interval (the |y_true - mean_pred|
score). This is a deliberate methodological choice, not a default.

Honesty constraints (see context doc §2):
    * Coverage is guaranteed only when calibration+test points are exchangeable
      (same topology family). It is NOT guaranteed under arbitrary distribution
      shift to unseen topology families — that must be tested separately, not
      assumed.
    * All functions here are pure numpy/torch-free so they are trivially
      unit-testable and auditable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

_EPS = 1e-9


def studentized_score(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
) -> np.ndarray:
    """Heteroskedasticity-aware nonconformity score.

    score_i = |y_true_i - mean_pred_i| / max(std_pred_i, eps)

    Uses the model's predicted std as the scale so large predicted uncertainty
    tolerates larger residual (wider interval) and vice-versa.
    """
    y_true = np.asarray(y_true, dtype=float)
    mean_pred = np.asarray(mean_pred, dtype=float)
    std_pred = np.asarray(std_pred, dtype=float)
    scale = np.maximum(np.abs(std_pred), _EPS)
    return np.abs(y_true - mean_pred) / scale


def split_conformal_quantile(
    calibration_scores: Sequence[float],
    alpha: float,
) -> float:
    """Standard split-conformal quantile over the calibration set.

    q = the ceil((n+1)(1-alpha))/n order statistic of the calibration scores,
    i.e. the smallest score such that the empirical coverage on calibration
    reaches at least (n-1)/n of the target. This is the conventional
    finite-sample-corrected quantile (split conformal, Vovk 2005 / Lei et al.
    2018).

    Args:
        calibration_scores: nonconformity scores on the calibration split.
        alpha: miscoverage level (e.g. 0.10 -> 90% nominal coverage).

    Returns:
        The conformal quantile q_hat (a single float).
    """
    n = len(calibration_scores)
    if n == 0:
        return float("inf")
    scores = np.sort(np.asarray(calibration_scores, dtype=float))
    q_index = int(np.ceil((n + 1) * (1.0 - alpha)))
    q_index = min(max(q_index, 1), n)
    return float(scores[q_index - 1])


def conformal_interval(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    q_hat: float,
) -> np.ndarray:
    """Form prediction intervals as mean_pred ± q_hat * std_pred.

    Returns an array of shape (n, 2) with columns [lower, upper].
    """
    mean_pred = np.asarray(mean_pred, dtype=float)
    std_pred = np.asarray(std_pred, dtype=float)
    half_width = q_hat * np.abs(std_pred)
    lo = mean_pred - half_width
    hi = mean_pred + half_width
    return np.stack([lo, hi], axis=1)


def empirical_coverage(lo: np.ndarray, hi: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of true values that fall within [lo, hi] (inclusive)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def coverage_by_group(
    lo: np.ndarray,
    hi: np.ndarray,
    y_true: np.ndarray,
    groups: Sequence,
) -> Dict[str, Dict[str, float]]:
    """Empirical coverage broken down by a grouping key (e.g. nrecon bucket).

    Returns dict keyed by group (str) -> {"n": count, "coverage": fraction}.
    This is the disaggregated check that can reveal per-topology miscalibration
    that a pooled number would hide.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    groups = np.asarray(list(groups))
    out: Dict[str, Dict[str, float]] = {}
    for g in sorted(set(groups), key=lambda x: (isinstance(x, str), str(x))):
        mask = np.asarray(groups == g)
        if mask.sum() == 0:
            continue
        out[str(g)] = {
            "n": int(mask.sum()),
            "coverage": empirical_coverage(lo[mask], hi[mask], y_true[mask]),
        }
    return out
