"""W04. Multilayer perceptron with an explicit forward and backward pass.

Interface (Appendix A). params is a dict {"W1", "b1", ..., "WL", "bL"}; W_l has shape
(n_in, n_out). The output layer is softmax; the loss is mean cross-entropy. The hidden
activation is your choice (state it in MODEL_LOG.md).
    init_params(sizes, seed) -> params       sizes = [d, h1, ..., K]
    forward(params, X) -> (P, cache)          P has shape (n, K), rows sum to 1
    loss(params, X, Y) -> float               Y is one-hot (n, K); mean cross-entropy
    backward(params, cache, Y) -> grads       same keys and shapes as params, gradient of loss
"""
import numpy as np


def init_params(sizes, seed=0):
    raise NotImplementedError


def forward(params, X):
    raise NotImplementedError


def loss(params, X, Y):
    raise NotImplementedError


def backward(params, cache, Y):
    raise NotImplementedError
