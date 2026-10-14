"""Metrics used across the semester (primary metric: macro-F1).

Evaluation Metrics for CO3117 ML Assignment
Primary metric: Macro-F1
Secondary: Accuracy, Confusion Matrix
"""

import numpy as np


def _inputs_and_labels(y_true, y_pred, n_classes=None, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels must be one-dimensional arrays of equal length.")
    if y_true.size == 0:
        raise ValueError("At least one sample is required.")
    if n_classes is not None and labels is not None:
        raise ValueError("Specify labels or n_classes, not both.")
    if labels is None:
        # Infer actual labels, not every integer from zero to the maximum label.
        labels = np.arange(n_classes) if n_classes is not None else np.union1d(y_true, y_pred)
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or len(np.unique(labels)) != len(labels):
        raise ValueError("labels must be a nonempty sequence of unique classes.")
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("Every observed label must occur in labels.")
    return y_true, y_pred, labels


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    y_true, y_pred, _ = _inputs_and_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes=None, *, labels=None):
    """
    Compute confusion matrix.

    Returns:
        cm: np.ndarray of shape (n_classes, n_classes)
            Rows/columns follow labels, or sorted observed labels by default.
            Legacy n_classes explicitly selects labels 0 through n_classes - 1.
    """
    y_true, y_pred, labels = _inputs_and_labels(y_true, y_pred, n_classes, labels)
    indices = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[indices[t], indices[p]] += 1
    return cm


def precision_recall_f1_per_class(y_true, y_pred, n_classes=None, *, labels=None):
    """
    Compute precision, recall, and F1 for each class.

    Returns:
        precision: np.ndarray of shape (n_classes,)
        recall: np.ndarray of shape (n_classes,)
        f1: np.ndarray of shape (n_classes,)
    """
    cm = confusion_matrix(y_true, y_pred, n_classes, labels=labels)
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


def macro_f1(y_true, y_pred, n_classes=None, *, labels=None):
    """
    Compute Macro-F1 score (primary metric for CO3117 assignment).

    Average over actual labels by default; use labels for a fixed class universe.
    """
    _, _, f1 = precision_recall_f1_per_class(y_true, y_pred, n_classes, labels=labels)
    return float(f1.mean())


def print_classification_report(y_true, y_pred, class_names=None, *, labels=None):
    """Print a formatted classification report."""
    y_true, y_pred, labels = _inputs_and_labels(y_true, y_pred, labels=labels)
    precision, recall, f1 = precision_recall_f1_per_class(y_true, y_pred, labels=labels)
    n_classes = len(precision)

    if class_names is None:
        class_names = [str(c) for c in labels]
    if len(class_names) != n_classes:
        raise ValueError("class_names must contain one name per label.")

    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

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
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    print("Sanity test:")
    print(f"  Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"  Macro-F1: {macro_f1(y_true, y_pred):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print()
    print_classification_report(y_true, y_pred, ["Class 0", "Class 1", "Class 2"])
