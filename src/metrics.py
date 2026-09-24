"""
metrics.py — Evaluation Metrics for CO3117 ML Assignment

Primary metric: Macro-F1
Secondary: Accuracy, Confusion Matrix
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Compute confusion matrix.

    Returns:
        cm: np.ndarray of shape (n_classes, n_classes)
            cm[i, j] = number of samples with true label i predicted as j
    """
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(y_true, y_pred, n_classes=None):
    """
    Compute precision, recall, and F1 for each class.

    Returns:
        precision: np.ndarray of shape (n_classes,)
        recall: np.ndarray of shape (n_classes,)
        f1: np.ndarray of shape (n_classes,)
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n_classes = cm.shape[0]

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    return precision, recall, f1


def macro_f1(y_true, y_pred, n_classes=None):
    """
    Compute Macro-F1 score (primary metric for CO3117 assignment).

    Macro-F1 = average of per-class F1 scores.
    """
    _, _, f1 = precision_recall_f1_per_class(y_true, y_pred, n_classes)
    return f1.mean()


def print_classification_report(y_true, y_pred, class_names=None):
    """Print a formatted classification report."""
    precision, recall, f1 = precision_recall_f1_per_class(y_true, y_pred)
    n_classes = len(precision)

    if class_names is None:
        class_names = [str(c) for c in range(n_classes)]

    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)

    cm = confusion_matrix(y_true, y_pred, n_classes)

    for c in range(n_classes):
        support = cm[c, :].sum()
        print(
            f"{class_names[c]:<25} {precision[c]:>10.4f} {recall[c]:>10.4f} "
            f"{f1[c]:>10.4f} {support:>10d}"
        )

    print("-" * 65)
    print(f"{'Macro avg':<25} {precision.mean():>10.4f} {recall.mean():>10.4f} {f1.mean():>10.4f} {len(y_true):>10d}")
    print(f"{'Accuracy':<25} {accuracy(y_true, y_pred):>10.4f}")


if __name__ == "__main__":
    # Quick sanity test
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    print("Sanity test:")
    print(f"  Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"  Macro-F1: {macro_f1(y_true, y_pred):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print()
    print_classification_report(y_true, y_pred, ["Class 0", "Class 1", "Class 2"])
