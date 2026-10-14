import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as reference_cm, f1_score

from src.metrics import accuracy, confusion_matrix, macro_f1, print_classification_report


@pytest.mark.parametrize("labels", [[1, 2, 3, 4, 5, 6], [0, 2, 7], [-1, 1], ["a", "b"]])
def test_nonzero_and_noncontiguous_labels_match_reference(labels):
    truth = np.repeat(labels, 3)
    pred = truth.copy()
    pred[0] = labels[-1]
    assert macro_f1(truth, pred) == pytest.approx(f1_score(truth, pred, average="macro"))
    assert np.array_equal(confusion_matrix(truth, pred), reference_cm(truth, pred))


def test_perfect_har_predictions_have_f1_one():
    assert macro_f1([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 1.0


def test_fixed_labels_include_absent_classes_and_preserve_order():
    truth, pred, labels = [1, 1, 2], [1, 2, 2], [3, 2, 1]
    assert macro_f1(truth, pred, labels=labels) == pytest.approx(
        f1_score(truth, pred, labels=labels, average="macro", zero_division=0))
    assert np.array_equal(confusion_matrix(truth, pred, labels=labels),
                          reference_cm(truth, pred, labels=labels))


def test_legacy_zero_based_n_classes():
    assert confusion_matrix([1], [1], n_classes=3).shape == (3, 3)
    assert macro_f1([1], [1], n_classes=3) == pytest.approx(1 / 3)


def test_report_supports_one_based_labels(capsys):
    print_classification_report([1, 2], [1, 2], ["Walk", "Sit"])
    output = capsys.readouterr().out
    assert "Walk" in output and "Sit" in output and "1.0000" in output


def test_accuracy_accepts_lists():
    assert accuracy([1, 2], [1, 1]) == 0.5


@pytest.mark.parametrize("truth,pred", [([], []), ([1, 2], [1]), ([[1]], [[1]])])
def test_invalid_arrays_are_rejected(truth, pred):
    with pytest.raises(ValueError):
        macro_f1(truth, pred)


def test_unknown_explicit_label_is_rejected():
    with pytest.raises(ValueError):
        confusion_matrix([1, 2], [1, 2], labels=[1])
