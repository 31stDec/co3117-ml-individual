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


def _logsumexp(a, axis=None, keepdims=False):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))   # per-class means
        self.var_ = np.zeros((n_classes, n_features))      # per-class variances
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0)                # ddof=0
            self.class_prior_[i] = X_c.shape[0] / X.shape[0]

        # Scale smoothing by the largest feature variance over all training rows.
        # This includes between-class variation, unlike max(per-class variance).
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += self.epsilon_
        return self

    def predict_log_proba(self, X):
        # Log prior
        log_prior = np.log(self.class_prior_)
        # Gaussian log-likelihood for each class
        # log N(x | mu, sigma^2) = -0.5 * [log(2*pi*sigma^2) + (x-mu)^2 / sigma^2]
        n_classes = len(self.classes_)
        jll = np.zeros((X.shape[0], n_classes))
        for i in range(n_classes):
            log_var = np.log(2 * np.pi * self.var_[i])
            diff = X - self.theta_[i]
            jll[:, i] = -0.5 * np.sum(log_var + diff ** 2 / self.var_[i], axis=1)
        jll += log_prior  # joint log-likelihood

        # Normalise to log-posterior via log-sum-exp
        log_sum = _logsumexp(jll, axis=1, keepdims=True)
        return jll - log_sum

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
