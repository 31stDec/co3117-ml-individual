"""
perceptron.py -- Depth-A From-Scratch Perceptron for CO3117

Implements:
  1. Binary Perceptron (classic sign-based update)
  2. One-vs-Rest (OvR) multi-class wrapper for 6-class UCI HAR

Theory-to-code mapping:
  - Forward pass:  z = w . x + b;  y_hat = sign(z)
  - Update rule:   if y_hat != y:  w += lr * y * x;  b += lr * y
  - Convergence:   guaranteed if data is linearly separable (Perceptron
                   Convergence Theorem); no guarantee otherwise.

Reference: Tom Mitchell (1997), Machine Learning, Section 4.5
"""

import numpy as np


class BinaryPerceptron:
    """
    Single-layer binary Perceptron with {-1, +1} labels.

    Parameters
    ----------
    learning_rate : float
        Step size for weight updates (eta).
    n_iters : int
        Maximum number of passes over the training set.
    random_seed : int or None
        Seed for reproducible weight initialisation and sample shuffling.
    """

    def __init__(self, learning_rate=0.01, n_iters=100, random_seed=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_seed = random_seed
        self.weights = None      # shape (n_features,)
        self.bias = None         # scalar
        self.errors_per_epoch = []  # for convergence analysis

    def _sign(self, z):
        """Hard threshold: returns +1 if z >= 0, else -1."""
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        """
        Train the perceptron on binary labels y in {-1, +1}.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — values in {-1, +1}
        """
        rng = np.random.RandomState(self.random_seed)
        n_samples, n_features = X.shape

        # Initialise weights to small random values (not zeros, to break
        # symmetry when features are correlated)
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.n_iters):
            # Shuffle training order each epoch to avoid cyclic stalls
            indices = rng.permutation(n_samples)
            n_errors = 0

            for i in indices:
                xi, yi = X[i], y[i]

                # Forward pass
                z = np.dot(self.weights, xi) + self.bias
                y_hat = self._sign(z)

                # Update only on misclassification
                if y_hat != yi:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
                    n_errors += 1

            self.errors_per_epoch.append(n_errors)

            # Early stop if perfectly separated
            if n_errors == 0:
                break

        return self

    def predict(self, X):
        """Return {-1, +1} predictions."""
        z = X @ self.weights + self.bias
        return self._sign(z)

    def decision_function(self, X):
        """Return raw signed distance to decision boundary."""
        return X @ self.weights + self.bias


class MultiClassPerceptron:
    """
    One-vs-Rest multi-class Perceptron.

    Trains K independent BinaryPerceptrons (one per class).
    At prediction time, picks the class whose perceptron returns the
    highest raw score (decision_function value).

    Parameters
    ----------
    learning_rate : float
    n_iters : int
    random_seed : int or None
    """

    def __init__(self, learning_rate=0.01, n_iters=100, random_seed=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_seed = random_seed
        self.classes_ = None
        self.classifiers_ = {}  # class_label -> BinaryPerceptron

    def fit(self, X, y):
        """
        Train one binary perceptron per class (One-vs-Rest).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — original class labels
        """
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            # Binarise: current class = +1, all others = -1
            y_binary = np.where(y == cls, 1, -1)

            clf = BinaryPerceptron(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                random_seed=self.random_seed,
            )
            clf.fit(X, y_binary)
            self.classifiers_[cls] = clf

        return self

    def predict(self, X):
        """
        Predict class labels by argmax over per-class decision scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        # Collect raw scores from each OvR classifier
        scores = np.column_stack([
            self.classifiers_[cls].decision_function(X)
            for cls in self.classes_
        ])
        # Pick the class with the highest score
        best_idx = np.argmax(scores, axis=1)
        return self.classes_[best_idx]

    def get_convergence_info(self):
        """Return per-class error curves for analysis."""
        return {
            cls: clf.errors_per_epoch
            for cls, clf in self.classifiers_.items()
        }


# === Quick sanity check on XOR (expected to fail) ===
if __name__ == "__main__":
    print("=" * 50)
    print("Sanity Check 1: Linearly Separable (AND gate)")
    print("=" * 50)

    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    p = BinaryPerceptron(learning_rate=0.1, n_iters=100)
    p.fit(X_and, y_and)
    preds = p.predict(X_and)
    print(f"  Predictions: {preds}  (expected [-1, -1, -1,  1])")
    print(f"  Correct: {np.all(preds == y_and)}")
    print(f"  Converged in {len(p.errors_per_epoch)} epochs")
    print(f"  Error curve: {p.errors_per_epoch}")

    print()
    print("=" * 50)
    print("Sanity Check 2: NOT Linearly Separable (XOR gate)")
    print("=" * 50)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([-1, 1, 1, -1])

    p2 = BinaryPerceptron(learning_rate=0.1, n_iters=20)
    p2.fit(X_xor, y_xor)
    preds2 = p2.predict(X_xor)
    print(f"  Predictions: {preds2}  (expected [-1,  1,  1, -1])")
    print(f"  Correct: {np.all(preds2 == y_xor)}")
    print(f"  Final epoch errors: {p2.errors_per_epoch[-1]} (should be > 0)")
