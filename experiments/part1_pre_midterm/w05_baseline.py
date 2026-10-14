"""
Baseline experiment: subject-held-out validation by default.
Use --evaluate-test only to correct the already-published historical test baselines.
This establishes the minimum performance bar for all subsequent models.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import ACTIVITY_LABELS
from src.metrics import macro_f1, accuracy, confusion_matrix
from experiments.part1_pre_midterm.common import LABELS, ROOT, evaluation_data, save_metric
import argparse
import json


def majority_class_baseline(y_train, y_test):
    """Predict the most frequent training class for all test samples."""
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    y_pred = np.full_like(y_test, majority_class)
    return y_pred, majority_class


def random_baseline(y_train, y_test, seed=42):
    """Predict random classes with training distribution."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()
    y_pred = rng.choice(classes, size=len(y_test), p=probs)
    return y_pred


def main():
    print("=" * 65)
    print("CO3117 — Baseline Experiments")
    print("Dataset: UCI HAR Using Smartphones")
    print("=" * 65)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true", help="Correct the historical test baselines")
    args = parser.parse_args()
    X_train, y_train, X_test, y_test, metadata = evaluation_data(args.evaluate_test)
    print("Evaluation split:", metadata["split"])
    class_names = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]

    print(f"\nTrain: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Evaluation: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes: {class_names}")

    # --- Baseline 1: Majority class ---
    print("\n" + "=" * 65)
    print("BASELINE 1: Majority-Class Classifier")
    print("=" * 65)
    y_pred_maj, maj_class = majority_class_baseline(y_train, y_test)
    print(f"Most frequent class in training: {maj_class} ({ACTIVITY_LABELS[maj_class]})")
    print(f"\nMacro-F1:  {macro_f1(y_test, y_pred_maj, labels=LABELS):.4f}")
    print(f"Accuracy:  {accuracy(y_test, y_pred_maj):.4f}")

    # --- Baseline 2: Random (proportional) ---
    print("\n" + "=" * 65)
    print("BASELINE 2: Random Classifier (training distribution)")
    print("=" * 65)
    y_pred_rnd = random_baseline(y_train, y_test)
    print(f"\nMacro-F1:  {macro_f1(y_test, y_pred_rnd, labels=LABELS):.4f}")
    print(f"Accuracy:  {accuracy(y_test, y_pred_rnd):.4f}")

    # --- Data summary ---
    print("\n" + "=" * 65)
    print("DATA SUMMARY")
    print("=" * 65)
    print("\nTraining class distribution:")
    for cls in sorted(ACTIVITY_LABELS.keys()):
        count = np.sum(y_train == cls)
        pct = count / len(y_train) * 100
        print(f"  {cls} ({ACTIVITY_LABELS[cls]:<22s}): {count:>5d} ({pct:5.1f}%)")

    print("\nEvaluation class distribution:")
    for cls in sorted(ACTIVITY_LABELS.keys()):
        count = np.sum(y_test == cls)
        pct = count / len(y_test) * 100
        print(f"  {cls} ({ACTIVITY_LABELS[cls]:<22s}): {count:>5d} ({pct:5.1f}%)")

    print("\nFeature statistics (train):")
    print(f"  Min:  {X_train.min():.4f}")
    print(f"  Max:  {X_train.max():.4f}")
    print(f"  Mean: {X_train.mean():.4f}")
    print(f"  Std:  {X_train.std():.4f}")

    # Preserve other model rows; keep validation and test in separate files.
    records = []
    for name, pred, note in [
        ("majority_baseline", y_pred_maj, f"Always predict {ACTIVITY_LABELS[maj_class]}"),
        ("random_baseline", y_pred_rnd, "Random with training distribution; seed=42"),
    ]:
        score, acc = macro_f1(y_test, pred, labels=LABELS), accuracy(y_test, pred)
        save_metric({"model": name, "chapter": 1, "depth": "baseline", "week": "W05",
                     "macro_f1": f"{score:.6f}", "accuracy": f"{acc:.6f}", "train_time_s": "0.0",
                     "notes": note + "; corrected six-class F1"}, metadata["split"])
        records.append({"model": name, "macro_f1": score, "accuracy": acc,
                        "confusion_matrix": confusion_matrix(y_test, pred, labels=LABELS).tolist()})
    (ROOT / "results" / f"baseline_{metadata['split']}_audit.json").write_text(
        json.dumps({"labels": LABELS.tolist(), "metadata": metadata, "runs": records}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
