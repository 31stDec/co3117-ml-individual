import json
from pathlib import Path

import numpy as np
from src.metrics import accuracy, confusion_matrix, macro_f1
from experiments.part1_pre_midterm.common import LABELS, ROOT, save_metric


def scores(y_train, pred_train, y_validation, pred_validation):
    cm = confusion_matrix(y_validation, pred_validation, labels=LABELS)
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    i, j = np.unravel_index(np.argmax(errors), errors.shape)
    return {"train_macro_f1": macro_f1(y_train, pred_train, labels=LABELS),
            "train_accuracy": accuracy(y_train, pred_train),
            "validation_macro_f1": macro_f1(y_validation, pred_validation, labels=LABELS),
            "validation_accuracy": accuracy(y_validation, pred_validation),
            "confusion_matrix": cm.tolist(),
            "largest_directed_confusion": {"true_label": int(LABELS[i]),
                                           "predicted_label": int(LABELS[j]), "count": int(errors[i, j])}}


def save_run_metric(record, chapter, depth, week, notes):
    save_metric({"model": record["model"], "chapter": chapter, "depth": depth, "week": week,
                 "macro_f1": f"{record['validation_macro_f1']:.6f}",
                 "accuracy": f"{record['validation_accuracy']:.6f}",
                 "train_time_s": f"{record['train_time_s']:.4f}", "notes": notes}, "validation")


def write_json(name, value):
    (ROOT / "results" / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def feature_names():
    # Some HAR feature names are repeated, so always record index and name together.
    path = ROOT / "data" / "raw" / "UCI HAR Dataset" / "features.txt"
    return [line.strip().split(maxsplit=1)[1] for line in path.read_text().splitlines()]
