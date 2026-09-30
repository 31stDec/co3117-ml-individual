import numpy as np

from src.from_scratch import perceptron as pc
from .conftest import call

C = "perceptron"


def test_update_rule_on_mistake_and_no_update_when_correct():
    w, b = np.array([0.0, 0.0]), 0.0
    w1, b1 = call(C, pc.perceptron_update, w.copy(), b, np.array([1.0, 2.0]), 1, 0.5)
    assert np.allclose(w1, [0.5, 1.0]) and np.isclose(b1, 0.5)
    w2, b2 = call(C, pc.perceptron_update, np.array([1.0, 1.0]), 0.0, np.array([1.0, 2.0]), 1, 0.5)
    assert np.allclose(w2, [1.0, 1.0]) and np.isclose(b2, 0.0)


def test_converges_on_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = np.where(X @ np.array([2.0, -1.0]) + 0.3 > 0, 1, -1)
    keep = np.abs(X @ np.array([2.0, -1.0]) + 0.3) > 0.2          # enforce a margin
    X, y = X[keep], y[keep]
    model = call(C, pc.Perceptron(lr=1.0, max_epochs=200).fit, X, y)
    assert np.all(model.predict(X) == y)


def test_predict_labels_are_signed():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]])
    y = np.array([1, 1, -1])
    model = call(C, pc.Perceptron(max_epochs=50).fit, X, y)
    assert set(np.unique(model.predict(X))) <= {-1, 1}
