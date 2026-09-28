"""
w05_perceptron.py -- W05 Perceptron Experiment on UCI HAR

Experiment design:
  1. Load UCI HAR (frozen data protocol)
  2. Standardise features (fit on train, transform test)
  3. Train OvR MultiClassPerceptron
  4. Evaluate: Macro-F1, Accuracy, per-class report
  5. Compare with baselines from w05_baseline.py
  6. Learning rate sensitivity analysis

Expected outcome: Perceptron should significantly beat
majority-class (0.044) and random (0.139) baselines.
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import get_train_test_data, ACTIVITY_LABELS
from src.metrics import macro_f1, accuracy, print_classification_report
from src.from_scratch.perceptron import MultiClassPerceptron

RANDOM_SEED = 42


def standardise(X_train, X_test):
    """Z-score normalisation: fit on train, transform both."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def run_experiment(learning_rate=0.01, n_iters=100):
    """Train and evaluate one Perceptron configuration."""
    # Load data
    X_train, y_train, X_test, y_test = get_train_test_data()

    # Standardise (critical for Perceptron convergence)
    X_train_s, X_test_s = standardise(X_train, X_test)

    # Train
    clf = MultiClassPerceptron(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_seed=RANDOM_SEED,
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    train_time_s = time.time() - t0

    # Evaluate
    y_pred_train = clf.predict(X_train_s)
    y_pred_test = clf.predict(X_test_s)

    train_f1 = macro_f1(y_train, y_pred_train)
    test_f1 = macro_f1(y_test, y_pred_test)
    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)

    return {
        "lr": learning_rate,
        "n_iters": n_iters,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time_s": train_time_s,
        "clf": clf,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }


def main():
    print("=" * 65)
    print("CO3117 W05 | Perceptron (OvR) on UCI HAR")
    print("=" * 65)
    print()

    # === Main experiment ===
    print("[1] Training Perceptron (lr=0.01, max_iters=100)...")
    result = run_experiment(learning_rate=0.01, n_iters=100)

    # Labels are 1-6, so build names array with index 0 as placeholder
    class_names = ["(unused)"] + [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]
    n_classes = max(result["y_test"].max(), result["y_pred_test"].max()) + 1

    print()
    print("--- Classification Report (Test Set) ---")
    # Print only classes 1-6
    from src.metrics import precision_recall_f1_per_class, confusion_matrix as cm_fn
    prec, rec, f1_arr = precision_recall_f1_per_class(
        result["y_test"], result["y_pred_test"], n_classes
    )
    cm = cm_fn(result["y_test"], result["y_pred_test"], n_classes)
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 65}")
    for c in range(1, n_classes):
        support = cm[c, :].sum()
        print(f"  {class_names[c]:<25} {prec[c]:>10.4f} {rec[c]:>10.4f} {f1_arr[c]:>10.4f} {support:>10d}")
    print(f"  {'-' * 65}")
    mf1_active = f1_arr[1:].mean()
    print(f"  {'Macro avg (1-6)':<25} {prec[1:].mean():>10.4f} {rec[1:].mean():>10.4f} {mf1_active:>10.4f} {len(result['y_test']):>10d}")

    print()
    print("--- Summary ---")
    print(f"  Train Macro-F1: {result['train_f1']:.4f}  |  Accuracy: {result['train_acc']:.4f}")
    print(f"  Test  Macro-F1: {result['test_f1']:.4f}  |  Accuracy: {result['test_acc']:.4f}")

    # === Comparison with baselines ===
    print()
    print("--- Comparison with Baselines ---")
    print(f"  {'Model':<30} {'Test Macro-F1':>15} {'Test Accuracy':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Majority-class':<30} {'0.0442':>15} {'0.1822':>15}")
    print(f"  {'Stratified random':<30} {'0.1388':>15} {'0.1659':>15}")
    print(f"  {'Perceptron (OvR)':<30} {result['test_f1']:>15.4f} {result['test_acc']:>15.4f}")

    # === Learning rate sensitivity ===
    print()
    print("[2] Learning Rate Sensitivity Analysis...")
    print(f"  {'LR':<10} {'Train F1':>10} {'Test F1':>10} {'Test Acc':>10}")
    print(f"  {'-'*40}")

    lr_results = []
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        r = run_experiment(learning_rate=lr, n_iters=100)
        lr_results.append(r)
        print(f"  {lr:<10.3f} {r['train_f1']:>10.4f} {r['test_f1']:>10.4f} {r['test_acc']:>10.4f}")

    # === Convergence info ===
    print()
    print("[3] Convergence (errors per epoch, class 1 detector)...")
    conv = result["clf"].get_convergence_info()
    class1_errors = conv[1]
    total_epochs = len(class1_errors)
    print(f"  Total epochs: {total_epochs}")
    print(f"  First 5 epoch errors: {class1_errors[:5]}")
    print(f"  Last 5 epoch errors:  {class1_errors[-5:]}")

    # === Append to metrics.csv ===
    metrics_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "metrics.csv"
    )
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(
            f"perceptron_ovr,4,Depth-A,W05,"
            f"{result['test_f1']:.4f},{result['test_acc']:.4f},"
            f"{result['train_time_s']:.2f},"
            f"OvR Perceptron lr={result['lr']} iters={result['n_iters']}\n"
        )
    print(f"\nResults appended to results/metrics.csv")
    print("Done.")


if __name__ == "__main__":
    main()
