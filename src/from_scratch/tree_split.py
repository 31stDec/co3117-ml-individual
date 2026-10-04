"""W03 (release catch-up). One impurity/split routine for decision trees.

Interface (Appendix A):
    entropy(y) -> float                 Shannon entropy in bits of the label array y.
    gini(y) -> float                    Gini impurity of y.
    information_gain(y, mask) -> float  Entropy of y minus the weighted entropy of y[mask], y[~mask].
    best_threshold(x, y) -> (t, gain)   Best split x <= t on one continuous feature by information
                                        gain; t is a midpoint between two consecutive distinct values.
                                        Return (None, 0.0) when x has a single distinct value.
"""

import numpy as np


def entropy(y):
    """Shannon entropy in bits of the label array y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(-np.sum(p * np.log2(p)))


def gini(y):
    """Gini impurity of y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


def information_gain(y, mask):
    """Entropy of y minus the weighted entropy of y[mask], y[~mask]."""
    n = len(y)
    if n == 0:
        return 0.0
    mask = np.asarray(mask, dtype=bool)
    n_left, n_right = np.sum(mask), np.sum(~mask)
    if n_left == 0 or n_right == 0:
        return 0.0
    h_parent = entropy(y)
    h_left = entropy(y[mask])
    h_right = entropy(y[~mask])
    return float(h_parent - (n_left / n * h_left + n_right / n * h_right))


def best_threshold(x, y):
    """
    Best split x <= t on one continuous feature by information gain;
    t is a midpoint between two consecutive distinct values.
    Return (None, 0.0) when x has a single distinct value.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    unique_x = np.unique(x)
    if len(unique_x) <= 1:
        return None, 0.0

    midpoints = (unique_x[:-1] + unique_x[1:]) / 2.0
    best_t = None
    best_gain = -1.0

    for t in midpoints:
        mask = (x <= t)
        gain = information_gain(y, mask)
        if gain > best_gain:
            best_gain = gain
            best_t = t

    return float(best_t) if best_t is not None else None, float(max(0.0, best_gain))
