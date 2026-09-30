"""W12. Principal component analysis.

Interface (Appendix A):
    fit_pca(X, k) -> (mean, components, explained_variance)
        mean (d,); components (k, d) with orthonormal rows, ordered by decreasing variance;
        explained_variance (k,) = sample variances (ddof = 1) of the projections.
    transform(X, mean, components) -> Z (n, k)
    inverse_transform(Z, mean, components) -> X_hat (n, d)
"""
import numpy as np


def fit_pca(X, k):
    raise NotImplementedError


def transform(X, mean, components):
    raise NotImplementedError


def inverse_transform(Z, mean, components):
    raise NotImplementedError
