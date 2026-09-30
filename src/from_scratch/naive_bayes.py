"""W05. Naive Bayes appropriate to your representation. Implement AT LEAST ONE of the two
classes below and leave the other raising NotImplementedError.

Interface (Appendix A):
    class GaussianNB(var_smoothing=1e-9)       continuous features; per-class variances use ddof = 0
                                               plus var_smoothing * (largest feature variance of X)
    class MultinomialNB(alpha=1.0)             count features, Laplace/Lidstone smoothing
        fit(X, y) -> self; attribute classes_ (sorted)
        predict_log_proba(X) -> (n, K) normalised log-posteriors, columns in classes_ order
        predict(X) -> labels from classes_
"""
import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
