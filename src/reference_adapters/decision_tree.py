"""Depth-B continuous classification tree adapted from ML-From-Scratch.

Reference: docs/reference/decision_tree.py (MIT, Erik Linder-Noren).
Retains recursive maximum-information-gain splitting and majority leaves.
Changes: vectorized midpoint search, <= routing, strict depth limit and minimum
child size. This is an attributed Codex-assisted adaptation, not a Depth-A claim.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    value: int
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


def _entropy_counts(counts):
    totals = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -(p * logp).sum(axis=-1)


class ContinuousTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, min_gain=1e-7):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if self.min_samples_leaf < 1 or (self.max_depth is not None and self.max_depth < 0):
            raise ValueError("Invalid tree stopping parameters")
        if len(X) != len(y) or not len(y) or not np.isfinite(X).all():
            raise ValueError("Finite, nonempty training data is required")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.depth_, self.n_leaves_ = 0, 0
        self.root_ = self._build(X, encoded, 0)
        total = self.feature_importances_.sum()
        if total:
            self.feature_importances_ /= total
        return self

    def _best_split(self, X, y):
        n, k = len(y), len(self.classes_)
        parent_counts = np.bincount(y, minlength=k)
        parent_entropy = _entropy_counts(parent_counts)
        best_gain, best_feature, best_threshold = self.min_gain, None, None
        identity = np.eye(k, dtype=int)
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            positions = np.arange(1, n)
            valid = ((values[:-1] < values[1:]) &
                     (positions >= self.min_samples_leaf) &
                     (n - positions >= self.min_samples_leaf))
            if not valid.any():
                continue
            positions = positions[valid]
            left_counts = np.cumsum(identity[y[order]], axis=0)[:-1][valid]
            right_counts = parent_counts - left_counts
            gain = parent_entropy - (positions * _entropy_counts(left_counts) +
                                     (n - positions) * _entropy_counts(right_counts)) / n
            candidate = int(np.argmax(gain))
            if gain[candidate] > best_gain:
                cut = positions[candidate]
                best_gain = float(gain[candidate])
                best_feature = feature
                best_threshold = float(values[cut - 1] + (values[cut] - values[cut - 1]) / 2)
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = Node(value=int(np.argmax(counts)))
        self.depth_ = max(self.depth_, depth)
        stop = (np.count_nonzero(counts) == 1 or len(y) < 2 * self.min_samples_leaf or
                (self.max_depth is not None and depth >= self.max_depth))
        if not stop:
            feature, threshold, gain = self._best_split(X, y)
            if feature is not None:
                mask = X[:, feature] <= threshold
                node.feature, node.threshold = feature, threshold
                self.feature_importances_[feature] += len(y) * gain
                node.left = self._build(X[mask], y[mask], depth + 1)
                node.right = self._build(X[~mask], y[~mask], depth + 1)
                return node
        self.n_leaves_ += 1
        return node

    def predict(self, X):
        output = []
        for x in np.asarray(X):
            node = self.root_
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            output.append(self.classes_[node.value])
        return np.asarray(output)
