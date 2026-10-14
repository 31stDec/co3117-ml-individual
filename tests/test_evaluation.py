import csv

import numpy as np

from experiments.part1_pre_midterm import common


def test_validation_uses_only_train_and_keeps_subjects_disjoint(monkeypatch):
    subjects = np.repeat(np.arange(12), 6)
    X = np.column_stack([subjects, np.arange(len(subjects))])
    y = np.tile(np.arange(1, 7), 12)

    def load(subset):
        assert subset == "train", "Default evaluation must not open test files"
        return X, y, subjects

    monkeypatch.setattr(common, "load_data", load)
    a = common.evaluation_data()
    b = common.evaluation_data()
    assert not set(a[4]["fit_subjects"]) & set(a[4]["evaluation_subjects"])
    assert len(a[1]) + len(a[3]) == len(y)
    assert a[4] == b[4]
    assert np.array_equal(a[0], b[0])


def test_scaler_uses_fit_rows_only():
    fit, evaluation = np.array([[0., 5.], [2., 5.]]), np.array([[100., 9.]])
    scaled_fit, scaled_evaluation = common.standardise(fit, evaluation)
    assert np.allclose(scaled_fit, [[-1, 0], [1, 0]])
    assert np.allclose(scaled_evaluation, [[99, 4]])


def test_saving_one_model_preserves_other_models_and_separates_splits(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(common, "ROOT", tmp_path)

    def row(name, score):
        return dict(zip(common.METRIC_FIELDS, [name, 1, "baseline", "W05", score, 0.5, 0, "test"]))

    common.save_metric(row("a", 0.1), "test")
    common.save_metric(row("b", 0.2), "test")
    common.save_metric(row("a", 0.3), "test")
    common.save_metric(row("a", 0.4), "validation")
    with (tmp_path / "results" / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {r["model"]: float(r["macro_f1"]) for r in rows} == {"a": 0.3, "b": 0.2}
    assert (tmp_path / "results" / "validation_metrics.csv").exists()
