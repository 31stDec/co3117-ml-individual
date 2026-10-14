"""Mini-batch SGD wrapper for the existing Depth-A MLP core.

Added with Codex assistance on 2026-10-14. Core forward/backward functions are
unchanged. RNG for initialization is separate from paired sample shuffling.
"""
import numpy as np
from src.from_scratch import mlp


class MLPClassifier:
    def __init__(self, hidden_width=32, lr=0.05, epochs=60, batch_size=128, seed=42):
        self.hidden_width, self.lr, self.epochs = hidden_width, lr, epochs
        self.batch_size, self.seed = batch_size, seed

    def fit(self, X, y, X_validation=None, y_validation=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[encoded]
        self.params_ = mlp.init_params([X.shape[1], self.hidden_width, len(self.classes_)], self.seed)
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        if X_validation is not None:
            mapping = {c: i for i, c in enumerate(self.classes_)}
            Yv = np.eye(len(self.classes_))[[mapping[c] for c in y_validation]]
        for epoch in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                batch = order[start:start + self.batch_size]
                _, cache = mlp.forward(self.params_, X[batch])
                grads = mlp.backward(self.params_, cache, Y[batch])
                for key in self.params_:
                    self.params_[key] -= self.lr * grads[key]
            record = {"epoch": epoch + 1, "train_loss": mlp.loss(self.params_, X, Y)}
            if X_validation is not None:
                record["validation_loss"] = mlp.loss(self.params_, X_validation, Yv)
            self.history_.append(record)
        return self

    def predict_proba(self, X):
        return mlp.forward(self.params_, X)[0]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
