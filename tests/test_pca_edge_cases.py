"""PCA invariants beyond the course's ordinary full-rank examples."""
import numpy as np
import pytest
from src.from_scratch.pca import fit_pca, transform, inverse_transform


def test_wide_rank_deficient_data_and_input_preservation():
    X = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.], [3., 6., 9., 12., 15.]])
    original = X.copy()
    mean, W, ev = fit_pca(X, 3)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(inverse_transform(transform(X, mean, W), mean, W), X, atol=1e-12)
    np.testing.assert_allclose(W @ W.T, np.eye(3), atol=1e-12)
    assert ev[0] > 0
    assert np.max(ev[1:]) < 1e-20


def test_constant_data_and_training_mean_for_new_rows():
    X = np.full((5, 3), 7.)
    mean, W, ev = fit_pca(X, 3)
    np.testing.assert_array_equal(ev, np.zeros(3))
    np.testing.assert_array_equal(transform(X, mean, W), np.zeros_like(X))
    new = np.array([[8., 5., 10.]])
    z = transform(new, mean, W)
    assert np.linalg.norm(z) > 0  # transforming one row must not center it on itself
    np.testing.assert_allclose(inverse_transform(z, mean, W), new)


@pytest.mark.parametrize('X,k', [([[1., 2.]], 1), ([[1., 2.], [3., 4.]], 0),
    ([[1., 2.], [3., 4.]], 3), ([[1., 2.], [3., 4.]], 1.5),
    ([[1., 2.], [3., 4.]], True), ([[1., np.nan], [3., 4.]], 1)])
def test_invalid_fit_inputs(X, k):
    with pytest.raises(ValueError):
        fit_pca(X, k)


def test_projection_shape_errors_and_empty_batches():
    mean, W, _ = fit_pca([[1., 2.], [3., 4.], [2., 1.]], 1)
    with pytest.raises(ValueError):
        transform([[1., 2., 3.]], mean, W)
    with pytest.raises(ValueError):
        inverse_transform([[1., 2.]], mean, W)
    assert transform(np.empty((0, 2)), mean, W).shape == (0, 1)
    assert inverse_transform(np.empty((0, 1)), mean, W).shape == (0, 2)
