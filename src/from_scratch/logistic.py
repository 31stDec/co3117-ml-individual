"""W15. Multinomial logistic (softmax) regression.

Interface (Appendix A):
    softmax(Z) -> P                        row-wise; must not overflow for logits of size 1e3
    softmax_loss_grad(W, b, X, y, reg=0.0) -> (loss, dW, db)
        W (d, K), b (K,), y integer labels in [0, K). loss = mean cross-entropy
        + (reg / 2) * sum(W ** 2); dW, db are its exact gradients.
    class SoftmaxRegression(lr=0.1, reg=0.0, max_iter=500, seed=0)
        fit(X, y) -> self; predict_proba(X) -> (n, K); predict(X) -> labels
"""
import numpy as np


def softmax(Z):
    raise NotImplementedError


def softmax_loss_grad(W, b, X, y, reg=0.0):
    raise NotImplementedError


class SoftmaxRegression:
    def __init__(self, lr=0.1, reg=0.0, max_iter=500, seed=0):
        self.lr, self.reg, self.max_iter, self.seed = lr, reg, max_iter, seed

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
