"""W09. Hidden Markov model core. Implement AT LEAST ONE of forward and viterbi.

Discrete HMM with N states and M symbols: pi (N,), A (N, N) with A[i, j] = P(s_t+1 = j | s_t = i),
B (N, M) with B[i, k] = P(o_t = k | s_t = i); obs is a sequence of ints in [0, M).
Interface (Appendix A):
    forward(pi, A, B, obs) -> (alpha, loglik)
        alpha (T, N): filtered probabilities P(s_t | o_1..o_t) (each row sums to 1);
        loglik = log P(o_1..o_T). Use scaling or log-space so that T = 1000 does not underflow.
    viterbi(pi, A, B, obs) -> (path, logprob)
        most probable state path (length T, ints) and log P(path, obs).
"""
import numpy as np


def _prepare_inputs(pi, A, B, obs):
    """Check the discrete HMM dimensions and probability distributions."""
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    obs = np.asarray(obs)
    if pi.ndim != 1 or pi.size == 0:
        raise ValueError("pi must be a nonempty vector")
    n = pi.size
    if A.shape != (n, n) or B.ndim != 2 or B.shape[0] != n or B.shape[1] == 0:
        raise ValueError("A must have shape (N, N) and B shape (N, M)")
    for name, values in (("pi", pi), ("A", A), ("B", B)):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} must contain finite nonnegative probabilities")
        if not np.allclose(values.sum(axis=-1), 1.0):
            raise ValueError(f"{name} distributions must sum to one")
    if obs.ndim != 1:
        raise ValueError("obs must be a one-dimensional sequence")
    if obs.size and (not np.issubdtype(obs.dtype, np.integer)
                     or np.any(obs < 0) or np.any(obs >= B.shape[1])):
        raise ValueError("observations must be integer symbol indices in [0, M)")
    return pi, A, B, obs.astype(int)


def forward(pi, A, B, obs):
    """Return filtered state probabilities and the sequence log-likelihood.

    Scale at each time step to avoid underflow. Unlike the unscaled forward
    values used in the lecture, each returned row sums to one. An impossible
    observation sequence raises ValueError because its conditional state
    probabilities are undefined. The empty sequence has log-likelihood zero.
    """
    pi, A, B, obs = _prepare_inputs(pi, A, B, obs)
    alpha = np.empty((len(obs), len(pi)))
    loglik = 0.0
    for t, symbol in enumerate(obs):
        predicted = pi if t == 0 else alpha[t - 1] @ A
        weights = predicted * B[:, symbol]
        scale = weights.sum()
        if scale == 0:
            raise ValueError("observation sequence has zero probability")
        alpha[t] = weights / scale
        loglik += np.log(scale)
    return alpha, float(loglik)


def viterbi(pi, A, B, obs):
    """Decode the most probable complete path using log-space and backtracking.

    Zero probabilities become negative infinity. Ties choose the lowest state
    index. An impossible sequence raises ValueError; an empty one returns an
    empty path with log-probability zero.
    """
    pi, A, B, obs = _prepare_inputs(pi, A, B, obs)
    T, N = len(obs), len(pi)
    if T == 0:
        return np.empty(0, dtype=int), 0.0
    with np.errstate(divide="ignore"):
        log_pi, log_A, log_B = np.log(pi), np.log(A), np.log(B)

    previous = np.zeros((T, N), dtype=int)
    scores = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        # Rows are previous states; columns are destination states.
        candidates = scores[:, None] + log_A
        previous[t] = np.argmax(candidates, axis=0)
        scores = candidates[previous[t], np.arange(N)] + log_B[:, obs[t]]

    path = np.empty(T, dtype=int)
    path[-1] = np.argmax(scores)
    logprob = float(scores[path[-1]])
    if np.isneginf(logprob):
        raise ValueError("observation sequence has zero probability")
    for t in range(T - 1, 0, -1):
        path[t - 1] = previous[t, path[t]]
    return path, logprob
