"""W03 / W05. Rosenblatt perceptron.

Depth-A from-scratch implementation for CO3117.

Interface (Appendix A):
    perceptron_update(w, b, x, y, lr) -> (w, b)
        One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
        w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    class Perceptron(lr=1.0, max_epochs=100, seed=0)
        fit(X, y) with y in {-1, +1}; returns self. Attributes w_, b_.
        predict(X) -> array of -1 / +1.
"""

import numpy as np


def perceptron_update(w, b, x, y, lr=1.0):
    """
    One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
    w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    """
    if y * (np.dot(w, x) + b) <= 0:
        w = w + lr * y * x
        b = b + lr * y
    return w, b


class Perceptron:
    """
    Rosenblatt binary Perceptron with {-1, +1} labels.

    Parameters
    ----------
    lr : float
        Learning rate (eta).
    max_epochs : int
        Maximum number of passes over the training set.
    seed : int or None
        Seed for reproducible weight initialization and sample shuffling.

    Attributes
    ----------
    w_ : np.ndarray, shape (n_features,)
        Learned weights.
    b_ : float
        Learned bias scalar.
    """

    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed
        self.w_ = None
        self.b_ = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        """
        Train binary Perceptron on y in {-1, +1}. Returns self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.max_epochs):
            n_errors = 0
            # Shuffle per epoch if random seed provided
            indices = rng.permutation(n_samples) if self.seed is not None else np.arange(n_samples)
            for i in indices:
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w_, xi) + self.b_) <= 0:
                    self.w_, self.b_ = perceptron_update(self.w_, self.b_, xi, yi, self.lr)
                    n_errors += 1
            self.errors_per_epoch.append(n_errors)
            if n_errors == 0:
                break
        return self

    def predict(self, X):
        """Return {-1, +1} predictions."""
        z = np.asarray(X, dtype=float) @ self.w_ + self.b_
        return np.where(z >= 0, 1, -1)

    def decision_function(self, X):
        """Return raw signed distance to decision boundary."""
        return np.asarray(X, dtype=float) @ self.w_ + self.b_


# Backward-compatible wrapper
class BinaryPerceptron(Perceptron):
    def __init__(self, learning_rate=0.01, n_iters=100, random_seed=42):
        super().__init__(lr=learning_rate, max_epochs=n_iters, seed=random_seed)
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_seed = random_seed

    @property
    def weights(self):
        return self.w_

    @property
    def bias(self):
        return self.b_


class MultiClassPerceptron:
    """
    One-vs-Rest multi-class Perceptron.
    Trains K independent BinaryPerceptrons (one per class).
    """

    def __init__(self, learning_rate=0.01, n_iters=100, random_seed=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_seed = random_seed
        self.classes_ = None
        self.classifiers_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            clf = Perceptron(
                lr=self.learning_rate,
                max_epochs=self.n_iters,
                seed=self.random_seed,
            )
            clf.fit(X, y_binary)
            self.classifiers_[cls] = clf
        return self

    def predict(self, X):
        scores = np.column_stack([
            self.classifiers_[cls].decision_function(X)
            for cls in self.classes_
        ])
        best_idx = np.argmax(scores, axis=1)
        return self.classes_[best_idx]

    def get_convergence_info(self):
        return {
            cls: clf.errors_per_epoch
            for cls, clf in self.classifiers_.items()
        }


# === Quick sanity check ===
if __name__ == "__main__":
    print("=" * 50)
    print("Sanity Check 1: Linearly Separable (AND gate)")
    print("=" * 50)

    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    p = Perceptron(lr=0.1, max_epochs=100)
    p.fit(X_and, y_and)
    preds = p.predict(X_and)
    print(f"  Predictions: {preds}  (expected [-1, -1, -1,  1])")
    print(f"  Correct: {np.all(preds == y_and)}")
    print(f"  Converged in {len(p.errors_per_epoch)} epochs")

    print()
    print("=" * 50)
    print("Sanity Check 2: NOT Linearly Separable (XOR gate)")
    print("=" * 50)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([-1, 1, 1, -1])

    p2 = Perceptron(lr=0.1, max_epochs=20)
    p2.fit(X_xor, y_xor)
    preds2 = p2.predict(X_xor)
    print(f"  Predictions: {preds2}  (expected [-1,  1,  1, -1])")
    print(f"  Correct: {np.all(preds2 == y_xor)}")
    print(f"  Final epoch errors: {p2.errors_per_epoch[-1]} (should be > 0)")
