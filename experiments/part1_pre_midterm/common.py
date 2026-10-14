"""Evaluation helpers: tuning uses subjects from the original training set only."""
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.data import ACTIVITY_LABELS, load_data

ROOT = Path(__file__).resolve().parents[2]
LABELS = np.array(sorted(ACTIVITY_LABELS))
METRIC_FIELDS = ["model", "chapter", "depth", "week", "macro_f1", "accuracy", "train_time_s", "notes"]


def evaluation_data(evaluate_test=False, seed=42):
    X, y, subjects = load_data("train")
    if evaluate_test:
        Xe, ye, eval_subjects = load_data("test")
        fit_subjects, split = subjects, "test"
    else:
        fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                          random_state=seed).split(X, y, subjects))
        Xe, ye, eval_subjects = X[val], y[val], subjects[val]
        X, y, fit_subjects = X[fit], y[fit], subjects[fit]
        split = "validation"
    assert not set(fit_subjects) & set(eval_subjects), "Subject leakage"
    assert set(y) == set(LABELS) and set(ye) == set(LABELS)
    metadata = {"split": split, "seed": seed,
                "fit_subjects": sorted(map(int, np.unique(fit_subjects))),
                "evaluation_subjects": sorted(map(int, np.unique(eval_subjects))),
                "fit_samples": len(y), "evaluation_samples": len(ye)}
    return X, y, Xe, ye, metadata


def standardise(X_fit, X_eval):
    mean, std = X_fit.mean(axis=0), X_fit.std(axis=0)
    std[std == 0] = 1.0
    return (X_fit - mean) / std, (X_eval - mean) / std


def save_metric(row, split):
    """Replace one model row and preserve other models' results."""
    path = ROOT / "results" / ("metrics.csv" if split == "test" else "validation_metrics.csv")
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    rows = [old for old in rows if old["model"] != row["model"]] + [row]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
