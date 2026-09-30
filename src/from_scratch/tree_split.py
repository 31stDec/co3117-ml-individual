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
    raise NotImplementedError


def gini(y):
    raise NotImplementedError


def information_gain(y, mask):
    raise NotImplementedError


def best_threshold(x, y):
    raise NotImplementedError
