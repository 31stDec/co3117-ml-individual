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
    """
    Initialize MLP weights and biases.
    Sizes = [d, h1, ..., K]. Returns dict of W1, b1, ..., WL, bL.
    """
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(sizes)):
        n_in, n_out = sizes[l - 1], sizes[l]
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (n_in + n_out))
        params[f"W{l}"] = rng.normal(0, scale, size=(n_in, n_out))
        params[f"b{l}"] = np.zeros(n_out, dtype=float)
    return params


def forward(params, X):
    """
    Forward pass through the network with tanh hidden activations and softmax output.
    Returns:
        P: softmax probabilities of shape (n, K), rows sum to 1
        cache: intermediate activations for backpropagation
    """
    X = np.asarray(X, dtype=float)
    cache = {"A0": X}
    L = len(params) // 2
    A = X

    for l in range(1, L):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        A = np.tanh(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    # Output layer: linear combination followed by numerically stable softmax
    ZL = A @ params[f"W{L}"] + params[f"b{L}"]
    shift_ZL = ZL - np.max(ZL, axis=1, keepdims=True)
    exp_ZL = np.exp(shift_ZL)
    P = exp_ZL / np.sum(exp_ZL, axis=1, keepdims=True)

    cache[f"Z{L}"] = ZL
    cache["P"] = P
    return P, cache


def loss(params, X, Y):
    """
    Mean cross-entropy loss. Y is one-hot of shape (n, K).
    """
    P, _ = forward(params, X)
    eps = 1e-15
    P_safe = np.clip(P, eps, 1.0)
    return float(-np.mean(np.sum(Y * np.log(P_safe), axis=1)))


def backward(params, cache, Y):
    """
    Backward pass computing gradients of mean cross-entropy loss wrt all weights and biases.
    Returns grads dict with exact same keys and shapes as params.
    """
    grads = {}
    L = len(params) // 2
    n = Y.shape[0]
    P = cache["P"]

    # Softmax cross-entropy derivative wrt pre-activation ZL
    dZ = (P - Y) / n

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        grads[f"W{l}"] = A_prev.T @ dZ
        grads[f"b{l}"] = np.sum(dZ, axis=0)

        if l > 1:
            dA = dZ @ params[f"W{l}"].T
            # tanh derivative: 1 - tanh^2(Z)
            dZ = dA * (1.0 - np.tanh(cache[f"Z{l - 1}"]) ** 2)

    return grads
