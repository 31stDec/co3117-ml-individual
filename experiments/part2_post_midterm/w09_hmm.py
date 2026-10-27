"""Run: python -m experiments.part2_post_midterm.w09_hmm

Validation only. Sequence boundaries use sensor overlap, never activity labels.
The only paired model change is independent vs learned state transitions.
"""
import ast
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data import ACTIVITY_LABELS, RAW_DIR, load_data
from src.from_scratch.hmm import forward, viterbi

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "w09_hmm"
SEEDS = [42, 0, 1, 2, 3]
SYMBOLS = 32


def reference_check():
    """Execute the pinned upstream class and helper, without package imports.

    AST extraction avoids importing unrelated numpy-ml modules. No function
    bodies are rewritten. Verify the saved source hashes before execution.
    """
    folder = ROOT / "docs/reference/numpy_ml_hmm"
    metadata = json.loads((folder / "provenance.json").read_text())
    namespace = {"np": np}
    for filename, name in (("misc.py", "logsumexp"), ("hmm.py", "MultinomialHMM")):
        content = (folder / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == metadata["files"][filename]["sha256"]
        tree = ast.parse(content)
        node = next(n for n in tree.body if getattr(n, "name", None) == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(folder / filename), "exec"), namespace)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pi = rng.dirichlet(np.ones(3))
        A = rng.dirichlet(np.ones(3), size=3)
        B = rng.dirichlet(np.ones(4), size=3)
        for length in (1, 6, 1000):
            obs = rng.integers(4, size=length)
            ref = namespace["MultinomialHMM"](A=A.copy(), B=B.copy(), pi=pi.copy())
            alpha, loglik = forward(pi, A, B, obs)
            path, score = viterbi(pi, A, B, obs)
            ref_alpha = ref._forward(obs).T
            ref_alpha = np.exp(ref_alpha - np.logaddexp.reduce(ref_alpha, axis=1)[:, None])
            ref_path, ref_score = ref.decode(obs)
            ref_ll = ref.log_likelihood(obs)
            assert np.allclose(alpha, ref_alpha, atol=1e-9, rtol=0)
            assert abs(loglik - ref_ll) < 1e-8
            assert abs(score - ref_score) < 1e-8
            # Different paths can tie (including roundoff in long recursions).
            # Compare both paths under the exact same, unsmoothed parameters.
            def path_score(q):
                q = np.asarray(q)
                return float(np.log(pi[q[0]]) + np.log(B[q, obs]).sum()
                             + np.log(A[q[:-1], q[1:]]).sum())
            gap = abs(path_score(path) - path_score(ref_path))
            assert gap < 1e-8
            rows.append({"seed": seed, "length": length,
                         "alpha_max_abs_error": float(np.max(np.abs(alpha-ref_alpha))),
                         "loglik_abs_error": abs(loglik-ref_ll),
                         "path_logprob_abs_error": abs(score-ref_score),
                         "path_equal": bool(np.array_equal(path, ref_path)),
                         "same_model_path_score_gap": gap})
    return {"source_commit": metadata["commit"], "cases": rows}


def continuous_segments(subjects):
    """Conservative chains: adjacent rows, same subject, matching 64 samples.

    All nine channels must match with atol=1e-6 and rtol=0. No sorting, joining
    across gaps, or label-based segmentation. Retain singleton windows too.
    """
    linked = subjects[:-1] == subjects[1:]
    hashes = {}
    for kind in ("total_acc", "body_acc", "body_gyro"):
        for axis in "xyz":
            path = RAW_DIR / f"UCI HAR Dataset/train/Inertial Signals/{kind}_{axis}_train.txt"
            values = np.loadtxt(path)
            assert values.shape == (len(subjects), 128)
            linked &= np.all(np.isclose(values[:-1, 64:], values[1:, :64], atol=1e-6, rtol=0), axis=1)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    segments = np.split(np.arange(len(subjects)), np.flatnonzero(~linked) + 1)
    assert np.array_equal(np.concatenate(segments), np.arange(len(subjects)))
    return segments, hashes


def estimate_model(labels, symbols, segments):
    """Supervised counts on fit subjects only, with fixed add-one smoothing."""
    pi = np.ones(6)
    A = np.ones((6, 6))
    B = np.ones((6, SYMBOLS))
    for seq in segments:
        q = labels[seq] - 1
        pi[q[0]] += 1
        np.add.at(A, (q[:-1], q[1:]), 1)
        np.add.at(B, (q, symbols[seq]), 1)
    return pi/pi.sum(), A/A.sum(axis=1, keepdims=True), B/B.sum(axis=1, keepdims=True)


def evaluate(labels, pred):
    return {"macro_f1": float(f1_score(labels, pred, labels=np.arange(1, 7), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
            "confusion_matrix": confusion_matrix(labels, pred, labels=np.arange(1, 7)).tolist()}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = reference_check()
    X, y, subjects = load_data("train")
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, subjects))
    fit_subjects, val_subjects = set(subjects[fit]), set(subjects[val])
    assert not fit_subjects & val_subjects
    segments, hashes = continuous_segments(subjects)
    for name in ("X_train.txt", "y_train.txt", "subject_train.txt"):
        path = RAW_DIR / "UCI HAR Dataset/train" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fit_seq = [s for s in segments if subjects[s[0]] in fit_subjects]
    val_seq = [s for s in segments if subjects[s[0]] in val_subjects]
    assert np.array_equal(np.sort(np.concatenate(fit_seq)), fit)
    assert np.array_equal(np.sort(np.concatenate(val_seq)), val)
    scaler = StandardScaler().fit(X[fit])
    Xfit, Xval = scaler.transform(X[fit]), scaler.transform(X[val])
    report = {"date": "2026-10-27", "split_seed": 42, "test_loaded": False,
              "fit_subjects": sorted(map(int, fit_subjects)), "validation_subjects": sorted(map(int, val_subjects)),
              "fit_windows": len(fit), "validation_windows": len(val),
              "symbols": SYMBOLS, "smoothing": 1.0, "seeds": SEEDS,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "signal_sha256": hashes, "reference_comparison": comparison, "sequence_audit": {}, "runs": []}
    for name, seqs in (("fit", fit_seq), ("validation", val_seq)):
        lengths = [len(s) for s in seqs]
        report["sequence_audit"][name] = {"sequences": len(seqs), "min_length": min(lengths),
            "median_length": float(np.median(lengths)), "max_length": max(lengths),
            "singletons": lengths.count(1), "verified_links": sum(lengths)-len(seqs),
            "label_changes_on_links": int(sum(np.sum(y[s[1:]] != y[s[:-1]]) for s in seqs))}
    prediction_rows = []
    for seed in SEEDS:
        quantizer = MiniBatchKMeans(n_clusters=SYMBOLS, random_state=seed, n_init=10, batch_size=1024)
        quantizer.fit(Xfit)
        symbols = np.full(len(y), -1, dtype=int)
        symbols[fit], symbols[val] = quantizer.predict(Xfit), quantizer.predict(Xval)
        pi, A, B = estimate_model(y, symbols, fit_seq)
        independent = np.argmax(pi[:, None] * B[:, symbols[val]], axis=0) + 1
        pred_filter, pred_viterbi = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
        iid_A = np.tile(pi, (6, 1))
        val_position = {int(row): k for k, row in enumerate(val)}
        for seq in val_seq:
            observations = symbols[seq]
            pred_filter[seq] = forward(pi, A, B, observations)[0].argmax(axis=1) + 1
            pred_viterbi[seq] = viterbi(pi, A, B, observations)[0] + 1
            # Replacing transition rows with pi must reproduce independent MAP.
            expected = independent[[val_position[int(row)] for row in seq]]
            assert np.array_equal(viterbi(pi, iid_A, B, observations)[0] + 1, expected)
        predictions = {"independent": independent, "filtering": pred_filter[val], "viterbi": pred_viterbi[val]}
        metrics = {name: evaluate(y[val], pred) for name, pred in predictions.items()}
        report["runs"].append({"seed": seed, "pi": pi.tolist(), "A": A.tolist(), "B": B.tolist(), "metrics": metrics,
            "fixed_by_viterbi": int(np.sum((independent != y[val]) & (pred_viterbi[val] == y[val]))),
            "harmed_by_viterbi": int(np.sum((independent == y[val]) & (pred_viterbi[val] != y[val])))})
        for seq_id, seq in enumerate(val_seq):
            for row in seq:
                prediction_rows.append({"seed": seed, "source_row_1based": int(row+1), "subject": int(subjects[row]),
                    "sequence": seq_id, "true": int(y[row]), "symbol": int(symbols[row]),
                    "independent": int(independent[val_position[int(row)]]),
                    "filtering": int(pred_filter[row]), "viterbi": int(pred_viterbi[row])})
        print(seed, {name: round(m["macro_f1"], 4) for name, m in metrics.items()}, flush=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    lines = ["# W09 HAR validation experiment", "", "Run: `python -m experiments.part2_post_midterm.w09_hmm`", "",
             "Fixed subject split; official test files are not loaded. Quantizer seeds vary; the split remains fixed.", "",
             "| Seed | Model | Macro-F1 | Accuracy |", "|---|---|---:|---:|"]
    for r in report["runs"]:
        for name, m in r["metrics"].items():
            lines.append(f"| {r['seed']} | {name} | {m['macro_f1']:.6f} | {m['accuracy']:.6f} |")
    lines.extend(["", "## Across quantizer seeds", "", "Values are mean +/- population SD; these are not independent subject splits or confidence intervals.", ""])
    for name in predictions:
        scores = np.array([[r["metrics"][name][m] for m in ("macro_f1", "accuracy")] for r in report["runs"]])
        lines.append(f"- {name}: Macro-F1 {scores[:,0].mean():.6f} +/- {scores[:,0].std():.6f}; accuracy {scores[:,1].mean():.6f} +/- {scores[:,1].std():.6f}.")
    lines.extend(["", "## Sequence audit", "", "```json", json.dumps(report["sequence_audit"], indent=2), "```", "",
        "## Failure examples (seed 42)", "", "First five remaining Viterbi errors in original row order, not selected for appearance.", "",
        "| Source row | Subject | True activity | Independent | Viterbi |", "|---|---|---|---|---|"])
    errors = [r for r in prediction_rows if r["seed"] == 42 and r["true"] != r["viterbi"]][:5]
    for r in errors:
        lines.append(f"| {r['source_row_1based']} | {r['subject']} | {ACTIVITY_LABELS[r['true']]} | {ACTIVITY_LABELS[r['independent']]} | {ACTIVITY_LABELS[r['viterbi']]} |")
    lines.extend(["", "## Interpretation limits", "",
        "Sequence boundaries come only from matching sensor overlap and subject identity. Labels are inspected afterward for this audit, not used to choose boundaries.", "",
        "Verified chains contain no observed activity changes. This evaluates within-activity denoising, not transition detection. The smoothed off-diagonal transition probabilities come from the add-one prior, not observed transitions. Overlapping windows also violate strict conditional emission independence, so likelihoods are not calibrated evidence for independent samples.", "",
        "Viterbi uses the full segment and is offline; filtering uses only the prefix. Quantization loses feature information. This comparison isolates sequence inference within one categorical-emission model, not superiority over all existing HAR classifiers.", "",
        "See audit.json for all confusion matrices, parameters, input hashes, and the pinned numpy-ml numerical comparison; predictions.csv contains every validation prediction."])
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
