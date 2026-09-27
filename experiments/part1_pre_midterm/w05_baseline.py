"""
Baseline experiment: Majority-class classifier + StandardScaler + basic analysis.
This establishes the minimum performance bar for all subsequent models.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import get_train_test_data, ACTIVITY_LABELS
from src.metrics import macro_f1, accuracy, print_classification_report, confusion_matrix


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

    # Load data
    X_train, y_train, X_test, y_test = get_train_test_data()
    class_names = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]

    print(f"\nTrain: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes: {class_names}")

    # --- Baseline 1: Majority class ---
    print("\n" + "=" * 65)
    print("BASELINE 1: Majority-Class Classifier")
    print("=" * 65)
    y_pred_maj, maj_class = majority_class_baseline(y_train, y_test)
    print(f"Most frequent class in training: {maj_class} ({ACTIVITY_LABELS[maj_class]})")
    print(f"\nMacro-F1:  {macro_f1(y_test, y_pred_maj, n_classes=7):.4f}")
    print(f"Accuracy:  {accuracy(y_test, y_pred_maj):.4f}")

    # --- Baseline 2: Random (proportional) ---
    print("\n" + "=" * 65)
    print("BASELINE 2: Random Classifier (training distribution)")
    print("=" * 65)
    y_pred_rnd = random_baseline(y_train, y_test)
    print(f"\nMacro-F1:  {macro_f1(y_test, y_pred_rnd, n_classes=7):.4f}")
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

    print("\nTest class distribution:")
    for cls in sorted(ACTIVITY_LABELS.keys()):
        count = np.sum(y_test == cls)
        pct = count / len(y_test) * 100
        print(f"  {cls} ({ACTIVITY_LABELS[cls]:<22s}): {count:>5d} ({pct:5.1f}%)")

    print("\nFeature statistics (train):")
    print(f"  Min:  {X_train.min():.4f}")
    print(f"  Max:  {X_train.max():.4f}")
    print(f"  Mean: {X_train.mean():.4f}")
    print(f"  Std:  {X_train.std():.4f}")

    # --- Save results ---
    results_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "metrics.csv")
    with open(results_path, "w") as f:
        f.write("model,chapter,depth,week,macro_f1,accuracy,train_time_s,notes\n")
        f.write(f"majority_baseline,1,baseline,W05,"
                f"{macro_f1(y_test, y_pred_maj, n_classes=7):.4f},"
                f"{accuracy(y_test, y_pred_maj):.4f},0.0,"
                f"Always predict {ACTIVITY_LABELS[maj_class]}\n")
        f.write(f"random_baseline,1,baseline,W05,"
                f"{macro_f1(y_test, y_pred_rnd, n_classes=7):.4f},"
                f"{accuracy(y_test, y_pred_rnd):.4f},0.0,"
                f"Random with training distribution\n")

    print(f"\n[OK] Results saved to {results_path}")


if __name__ == "__main__":
    main()
