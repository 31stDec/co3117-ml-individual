import numpy as np
import pytest

from src.from_scratch import tree_split as ts
from .conftest import call

C = "tree_split"


def test_entropy_and_gini_known_values():
    y = np.array([0, 0, 1, 1])
    assert call(C, ts.entropy, y) == pytest.approx(1.0)
    assert call(C, ts.gini, y) == pytest.approx(0.5)
    assert call(C, ts.entropy, np.array([2, 2, 2])) == pytest.approx(0.0)
    y3 = np.array([0, 1, 2])
    assert call(C, ts.entropy, y3) == pytest.approx(np.log2(3))
    assert call(C, ts.gini, y3) == pytest.approx(2 / 3)


def test_information_gain_perfect_and_useless_split():
    y = np.array([0, 0, 1, 1])
    assert call(C, ts.information_gain, y, np.array([True, True, False, False])) == pytest.approx(1.0)
    assert call(C, ts.information_gain, y, np.array([True, False, True, False])) == pytest.approx(0.0)


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40).round(2)
    y = (x + rng.normal(scale=0.5, size=40) > 0).astype(int)
    t, gain = call(C, ts.best_threshold, x, y)

    def H(v):
        _, c = np.unique(v, return_counts=True)
        p = c / c.sum()
        return -(p * np.log2(p)).sum()

    u = np.unique(x)
    best = max(H(y) - (np.mean(x <= m) * H(y[x <= m]) + np.mean(x > m) * H(y[x > m]))
               for m in (u[:-1] + u[1:]) / 2)
    assert gain == pytest.approx(best, abs=1e-9)
    assert H(y) - (np.mean(x <= t) * H(y[x <= t]) + np.mean(x > t) * H(y[x > t])) == pytest.approx(best, abs=1e-9)


def test_best_threshold_constant_feature():
    t, gain = call(C, ts.best_threshold, np.ones(5), np.array([0, 1, 0, 1, 1]))
    assert t is None and gain == 0.0
