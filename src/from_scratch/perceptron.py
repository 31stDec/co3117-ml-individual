"""W03. Rosenblatt perceptron.

Interface (Appendix A):
    perceptron_update(w, b, x, y, lr) -> (w, b)
        One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
        w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    class Perceptron(lr=1.0, max_epochs=100, seed=0)
        fit(X, y) with y in {-1, +1}; returns self. Attributes w_, b_.
        predict(X) -> array of -1 / +1.
"""
import numpy as np


def perceptron_update(w, b, x, y, lr):
    raise NotImplementedError


class Perceptron:
    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
