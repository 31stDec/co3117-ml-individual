import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB as Reference

from src.from_scratch.naive_bayes import GaussianNB


def test_smoothing_includes_between_class_variance():
    # Large mean separation makes global and within-class variances very different.
    X = np.array([[0., 0.], [1., 0.], [100., 0.], [101., 0.]])
    y = np.array([1, 1, 2, 2])
    own = GaussianNB(var_smoothing=0.1).fit(X, y)
    ref = Reference(var_smoothing=0.1).fit(X, y)
    assert own.epsilon_ == pytest.approx(0.1 * X.var(axis=0).max())
    assert np.allclose(own.var_, ref.var_)
    probes = np.array([[0., 0.], [30., 1.], [50., 0.], [100., 0.]])
    assert np.allclose(own.predict_log_proba(probes), ref.predict_log_proba(probes), atol=1e-10)
