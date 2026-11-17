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
    """Fit centered PCA by thin SVD; k is an integer in [1, min(n, d)].

    No scaling or whitening is applied. Variances use n-1 as required by
    the course API. Constant/rank-deficient inputs may have zero variances.
    """
    X = _matrix(X, "X")
    n, d = X.shape
    if n < 2 or d == 0:
        raise ValueError("PCA requires at least two rows and one feature")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 1 <= k <= min(n, d):
        raise ValueError("k must be between 1 and min(n_samples, n_features)")
    mean = X.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(X - mean, full_matrices=False)
    components = vt[:k].copy()
    # Eigenvector signs are arbitrary. Choose a stable convention for output.
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(k), pivots])
    components *= signs[:, None]
    return mean, components, singular_values[:k] ** 2 / (n - 1)


def transform(X, mean, components):
    """Project using the training mean and row-oriented components."""
    X = _matrix(X, "X")
    mean, components = _parameters(mean, components)
    if X.shape[1] != mean.size:
        raise ValueError("X feature count does not match mean")
    return (X - mean) @ components.T


def inverse_transform(Z, mean, components):
    """Reconstruct in original feature units, restoring the training mean."""
    Z = _matrix(Z, "Z")
    mean, components = _parameters(mean, components)
    if Z.shape[1] != components.shape[0]:
        raise ValueError("Z column count does not match components")
    return Z @ components + mean


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return value


def _parameters(mean, components):
    if np.iscomplexobj(mean):
        raise ValueError("mean must be real")
    mean = np.asarray(mean, dtype=float)
    components = _matrix(components, "components")
    if (mean.ndim != 1 or mean.size == 0 or not np.isfinite(mean).all()
            or components.shape[1] != mean.size or components.shape[0] == 0):
        raise ValueError("mean and components have incompatible shapes or values")
    return mean, components
