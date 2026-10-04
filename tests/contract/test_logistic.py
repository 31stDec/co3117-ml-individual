import numpy as np
import pytest

from src.from_scratch import logistic as lg
from .conftest import call

C = "logistic"


def test_softmax_is_stable():
    P = call(C, lg.softmax, np.array([[1000.0, 999.0, -1000.0], [0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(P))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[1], 1 / 3)


@pytest.mark.parametrize("reg", [0.0, 0.3])
def test_gradient_matches_numerical(reg):
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(15, 4)), rng.integers(0, 3, 15)
    W, b = rng.normal(size=(4, 3)), rng.normal(size=3)
    loss, dW, db = call(C, lg.softmax_loss_grad, W, b, X, y, reg)
    Z = X @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    ref = -np.mean(Z[np.arange(15), y] - np.log(np.exp(Z).sum(axis=1))) + reg / 2 * np.sum(W ** 2)
    assert loss == pytest.approx(ref, rel=1e-8)
    eps = 1e-6
    for P, G in ((W, dW), (b, db)):
        num = np.zeros_like(P)
        for i in np.ndindex(P.shape):
            old = P[i]
            P[i] = old + eps
            lp = lg.softmax_loss_grad(W, b, X, y, reg)[0]
            P[i] = old - eps
            lm = lg.softmax_loss_grad(W, b, X, y, reg)[0]
            P[i] = old
            num[i] = (lp - lm) / (2 * eps)
        assert np.allclose(G, num, atol=1e-6)


def test_fit_separates_blobs():
    rng = np.random.default_rng(4)
    centers = np.array([[0, 4], [4, 0], [-4, -4]])
    y = rng.integers(0, 3, 300)
    X = centers[y] + rng.normal(size=(300, 2))
    model = call(C, lg.SoftmaxRegression(lr=0.1, max_iter=500).fit, X, y)
    P = model.predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) > 0.95
