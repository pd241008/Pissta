import math

from timing.clark_max import clark_max


def test_clark_max_identical():
    mu, var = clark_max(1.0, 1.0, 1.0, 1.0, 0.0)
    assert 1.5 < mu < 1.7
    assert var > 0.0


def test_clark_max_zero_variance():
    mu, var = clark_max(2.0, 0.0, 1.0, 0.0, 0.0)
    assert mu > 1.0
    assert var >= 0.0


def test_clark_max_symmetry():
    mu1, var1 = clark_max(1.0, 1.0, 2.0, 1.0, 0.0)
    mu2, var2 = clark_max(2.0, 1.0, 1.0, 1.0, 0.0)
    assert math.isclose(mu1, mu2, rel_tol=1e-6)
    assert math.isclose(var1, var2, rel_tol=1e-6)
