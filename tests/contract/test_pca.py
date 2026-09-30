import numpy as np
import pytest

from src.from_scratch import pca
from .conftest import call

C = "pca"


def _data():
    rng = np.random.default_rng(7)
    L = rng.normal(size=(5, 5)) * np.array([5, 3, 1, 0.5, 0.1])
    return rng.normal(size=(120, 5)) @ L.T + np.array([1, -2, 0, 3, 5])


def test_components_and_variance_match_svd():
    X = _data()
    mean, W, ev = call(C, pca.fit_pca, X, 3)
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    assert np.allclose(mean, X.mean(axis=0))
    assert W.shape == (3, 5)
    assert np.allclose(W @ W.T, np.eye(3), atol=1e-8)
    assert np.allclose(np.abs(W @ Vt[:3].T), np.eye(3), atol=1e-6)   # equal up to sign
    assert np.allclose(ev, s[:3] ** 2 / (len(X) - 1), rtol=1e-6)


def test_projection_is_decorrelated_and_reconstruction_optimal():
    X = _data()
    mean, W, ev = call(C, pca.fit_pca, X, 2)
    Z = call(C, pca.transform, X, mean, W)
    C_z = np.cov(Z, rowvar=False)
    assert np.allclose(C_z, np.diag(ev), atol=1e-6)
    Xh = call(C, pca.inverse_transform, Z, mean, W)
    _, s, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    assert np.sum((X - Xh) ** 2) == pytest.approx(np.sum(s[2:] ** 2), rel=1e-6)
