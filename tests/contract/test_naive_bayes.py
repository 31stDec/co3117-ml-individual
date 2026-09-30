import numpy as np
import pytest

from src.from_scratch import naive_bayes as nb
from .conftest import implemented, is_required

C = "naive_bayes"


def _gauss():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(1.5, 0.7, (80, 3)), rng.normal(-1, 2, (80, 3))])
    y = np.repeat(["a", "b", "c"], 80)
    return X, y


def _counts():
    rng = np.random.default_rng(6)
    p = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]])
    y = rng.integers(0, 2, 150)
    X = np.array([rng.multinomial(20, p[c]) for c in y])
    return X, y


def _has(cls, data):
    X, y = data
    return implemented(cls().fit, X, y)


def test_at_least_one_variant_implemented():
    g, m = _has(nb.GaussianNB, _gauss()), _has(nb.MultinomialNB, _counts())
    if not (g or m):
        if is_required(C):
            pytest.fail("naive_bayes: implement GaussianNB or MultinomialNB")
        pytest.skip("naive_bayes: not implemented yet")


def test_gaussian_matches_sklearn():
    X, y = _gauss()
    if not _has(nb.GaussianNB, (X, y)):
        pytest.skip("GaussianNB not implemented")
    from sklearn.naive_bayes import GaussianNB as Ref
    own, ref = nb.GaussianNB().fit(X, y), Ref().fit(X, y)
    assert list(own.classes_) == list(ref.classes_)
    L = own.predict_log_proba(X)
    assert np.allclose(np.exp(L).sum(axis=1), 1.0)
    assert np.allclose(L, ref.predict_log_proba(X), atol=1e-4)
    assert np.all(own.predict(X) == ref.predict(X))


def test_multinomial_matches_sklearn():
    X, y = _counts()
    if not _has(nb.MultinomialNB, (X, y)):
        pytest.skip("MultinomialNB not implemented")
    from sklearn.naive_bayes import MultinomialNB as Ref
    own, ref = nb.MultinomialNB(alpha=1.0).fit(X, y), Ref(alpha=1.0).fit(X, y)
    assert np.allclose(own.predict_log_proba(X), ref.predict_log_proba(X), atol=1e-6)
    assert np.all(own.predict(X) == ref.predict(X))
