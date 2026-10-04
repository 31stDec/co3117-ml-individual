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


def forward(pi, A, B, obs):
    raise NotImplementedError


def viterbi(pi, A, B, obs):
    raise NotImplementedError
