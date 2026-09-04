import numpy as np
import pytest

from gnn_baseline.conformal import (
    conformal_interval,
    coverage_by_group,
    empirical_coverage,
    interval_width_stats,
    split_conformal_quantile,
    studentized_score,
)


def test_studentized_score_scales_by_std():
    y_true = np.array([10.0])
    mean_pred = np.array([9.0])
    std_pred = np.array([0.5])
    # residual 1.0 / scale 0.5 = 2.0
    assert studentized_score(y_true, mean_pred, std_pred)[0] == pytest.approx(2.0)


def test_studentized_score_larger_std_tolerates_larger_residual():
    r1 = studentized_score(
        np.array([10.0]), np.array([9.0]), np.array([1.0])
    )[0]
    r2 = studentized_score(
        np.array([10.0]), np.array([9.0]), np.array([2.0])
    )[0]
    assert r1 == pytest.approx(1.0)
    assert r2 == pytest.approx(0.5)
    assert r2 < r1


def test_studentized_score_zero_std_guarded():
    # Zero predicted std must not produce division by zero / inf.
    s = studentized_score(np.array([10.0]), np.array([9.0]), np.array([0.0]))
    assert np.isfinite(s[0])


def test_split_conformal_quantile_matches_manual_order_statistic():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=100)
    alpha = 0.10
    q = split_conformal_quantile(scores, alpha)
    n = len(scores)
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    q_idx = int(np.ceil((n + 1) * (1.0 - alpha)))
    q_idx = min(max(q_idx, 1), n)
    assert q == pytest.approx(float(sorted_scores[q_idx - 1]))


def test_split_conformal_quantile_nominal_coverage_on_calibration():
    # On the calibration set itself, coverage of [mean - q*std, mean + q*std]
    # for a Gaussian with std=1 must be >= 1-alpha (it's ~1-alpha on cal).
    rng = np.random.default_rng(1)
    mean_pred = rng.normal(size=300)
    y_true = mean_pred + rng.normal(size=300)  # residual ~ N(0,1)
    std_pred = np.ones(300)
    scores = studentized_score(y_true, mean_pred, std_pred)
    q = split_conformal_quantile(scores, alpha=0.10)
    lo, hi = conformal_interval(mean_pred, std_pred, q).T
    cov = empirical_coverage(lo, hi, y_true)
    # Finite-sample: calibration coverage is >= (n-1)/n * (1-alpha) approx 90%.
    assert cov >= 0.88


def test_empirical_coverage_perfect_and_zero():
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    y = np.array([5.0, 50.0])
    assert empirical_coverage(lo, hi, y) == pytest.approx(0.5)


def test_coverage_by_group_disaggregates():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 1.0, 10.0])
    y = np.array([5.0, 50.0, 5.0])
    groups = np.array([1, 1, 2])
    out = coverage_by_group(lo, hi, y, groups)
    assert out["1"]["n"] == 2
    assert out["1"]["coverage"] == pytest.approx(0.5)
    assert out["2"]["n"] == 1
    assert out["2"]["coverage"] == pytest.approx(1.0)


def test_heteroskedastic_interval_width_follows_std():
    mean = np.array([5.0, 5.0])
    std = np.array([1.0, 3.0])
    q = 2.0
    intervals = conformal_interval(mean, std, q)
    w0 = intervals[0, 1] - intervals[0, 0]
    w1 = intervals[1, 1] - intervals[1, 0]
    assert w0 == pytest.approx(4.0)
    assert w1 == pytest.approx(12.0)


def test_interval_width_stats_pooled():
    lo = np.array([0.0, 2.0, 5.0])
    hi = np.array([10.0, 4.0, 15.0])
    widths = hi - lo  # [10, 2, 10]
    result = interval_width_stats(lo, hi)
    assert "pooled" in result
    assert result["pooled"]["n"] == 3
    assert result["pooled"]["mean_width"] == pytest.approx(np.mean(widths))
    assert result["pooled"]["median_width"] == pytest.approx(np.median(widths))


def test_interval_width_stats_by_group():
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 4.0, 6.0, 8.0])
    groups = np.array([1, 1, 2, 2])
    result = interval_width_stats(lo, hi, groups)
    assert "1" in result
    assert "2" in result
    assert "pooled" in result
    assert result["1"]["n"] == 2
    assert result["1"]["mean_width"] == pytest.approx(3.0)  # (2+4)/2
    assert result["2"]["n"] == 2
    assert result["2"]["mean_width"] == pytest.approx(7.0)  # (6+8)/2
    assert result["pooled"]["n"] == 4


def test_interval_width_stats_empty():
    lo = np.array([], dtype=float)
    hi = np.array([], dtype=float)
    result = interval_width_stats(lo, hi)
    assert result["pooled"]["n"] == 0
    assert np.isnan(result["pooled"]["mean_width"])
