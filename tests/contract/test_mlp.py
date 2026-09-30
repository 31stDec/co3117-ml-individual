import numpy as np
import pytest

from src.from_scratch import mlp
from .conftest import call

C = "mlp"


def _data(seed=1, n=12, d=4, K=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = np.eye(K)[rng.integers(0, K, size=n)]
    return X, Y


def test_forward_returns_probabilities():
    X, Y = _data()
    params = call(C, mlp.init_params, [4, 5, 3], 0)
    P, _ = call(C, mlp.forward, params, X)
    assert P.shape == (12, 3)
    assert np.allclose(P.sum(axis=1), 1.0) and np.all(P >= 0)


@pytest.mark.parametrize("sizes", [[4, 5, 3], [4, 6, 5, 3]])
def test_backward_matches_numerical_gradient(sizes):
    X, Y = _data()
    params = call(C, mlp.init_params, sizes, 0)
    params = {k: v.astype(float) for k, v in params.items()}
    _, cache = call(C, mlp.forward, params, X)
    grads = call(C, mlp.backward, params, cache, Y)
    assert set(grads) == set(params)
    eps = 1e-6
    for k, v in params.items():
        assert grads[k].shape == v.shape, k
        num = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = v[i]
            v[i] = old + eps
            lp = mlp.loss(params, X, Y)
            v[i] = old - eps
            lm = mlp.loss(params, X, Y)
            v[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.linalg.norm(grads[k] - num) / max(1e-12, np.linalg.norm(grads[k]) + np.linalg.norm(num))
        assert rel < 1e-5, f"{k}: relative error {rel:.2e}"
