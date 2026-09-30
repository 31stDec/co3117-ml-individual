import itertools

import numpy as np
import pytest

from src.from_scratch import hmm
from .conftest import implemented, is_required

C = "hmm"


def _model(seed, N=3, M=4):
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(N))
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(M), size=N)
    obs = rng.integers(0, M, size=6)
    return pi, A, B, obs


def _joint(pi, A, B, obs, path):
    p = pi[path[0]] * B[path[0], obs[0]]
    for t in range(1, len(obs)):
        p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
    return p


def _paths(N, T):
    return itertools.product(range(N), repeat=T)


def test_at_least_one_routine_implemented():
    args = _model(0)
    if not (implemented(hmm.forward, *args) or implemented(hmm.viterbi, *args)):
        if is_required(C):
            pytest.fail("hmm: implement forward or viterbi")
        pytest.skip("hmm: not implemented yet")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_against_enumeration(seed):
    pi, A, B, obs = _model(seed)
    if not implemented(hmm.forward, pi, A, B, obs):
        pytest.skip("forward not implemented")
    alpha, loglik = hmm.forward(pi, A, B, obs)
    N, T = len(pi), len(obs)
    total = sum(_joint(pi, A, B, obs, p) for p in _paths(N, T))
    assert loglik == pytest.approx(np.log(total), abs=1e-8)
    # filtered marginal at the last step
    last = np.zeros(N)
    for p in _paths(N, T):
        last[p[-1]] += _joint(pi, A, B, obs, p)
    assert np.allclose(alpha[-1], last / last.sum(), atol=1e-8)
    assert np.allclose(alpha.sum(axis=1), 1.0)


def test_forward_long_sequence_no_underflow():
    pi, A, B, _ = _model(4)
    obs = np.random.default_rng(9).integers(0, 4, size=1000)
    if not implemented(hmm.forward, pi, A, B, obs[:3]):
        pytest.skip("forward not implemented")
    _, loglik = hmm.forward(pi, A, B, obs)
    assert np.isfinite(loglik) and loglik < 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_viterbi_against_enumeration(seed):
    pi, A, B, obs = _model(seed)
    if not implemented(hmm.viterbi, pi, A, B, obs):
        pytest.skip("viterbi not implemented")
    path, logprob = hmm.viterbi(pi, A, B, obs)
    best = max(_paths(len(pi), len(obs)), key=lambda p: _joint(pi, A, B, obs, p))
    assert logprob == pytest.approx(np.log(_joint(pi, A, B, obs, best)), abs=1e-8)
    assert _joint(pi, A, B, obs, list(path)) == pytest.approx(_joint(pi, A, B, obs, best), rel=1e-9)
