"""Leakage and continuity checks for the W09 sequence experiment."""
import numpy as np

from experiments.part2_post_midterm import w09_hmm as experiment


def test_boundaries_use_all_channels_and_never_cross_subjects(tmp_path, monkeypatch):
    folder = tmp_path / "UCI HAR Dataset/train/Inertial Signals"
    folder.mkdir(parents=True)
    signals = np.arange(128)[None, :] + np.array([0, 64, 128, 1000, 1064])[:, None]
    for kind in ("total_acc", "body_acc", "body_gyro"):
        for axis in "xyz":
            np.savetxt(folder / f"{kind}_{axis}_train.txt", signals)
    monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
    subjects = np.array([1, 1, 2, 2, 2])
    seqs, _ = experiment.continuous_segments(subjects)
    assert [s.tolist() for s in seqs] == [[0, 1], [2], [3, 4]]
    # A mismatch in even the last channel must break the candidate link.
    signals[4, 0] += 1
    np.savetxt(folder / "body_gyro_z_train.txt", signals)
    seqs, _ = experiment.continuous_segments(subjects)
    assert [s.tolist() for s in seqs] == [[0, 1], [2], [3], [4]]


def test_parameter_counts_ignore_excluded_rows_and_segment_gaps():
    labels = np.array([1, 2, 6, 3])
    symbols = np.array([0, 1, 31, 2])
    seqs = [np.array([0, 1]), np.array([3])]
    first = experiment.estimate_model(labels, symbols, seqs)
    labels[2], symbols[2] = 999, 999
    second = experiment.estimate_model(labels, symbols, seqs)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    pi, A, B = first
    np.testing.assert_allclose(pi, np.array([2, 1, 2, 1, 1, 1])/8)
    np.testing.assert_allclose(A[0], np.array([1, 2, 1, 1, 1, 1])/7)
    # No transition may be counted across the excluded row or a segment gap.
    np.testing.assert_allclose(A[1:], 1/6)
    assert B[0, 0] == 2/33
