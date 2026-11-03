"""W10 linear SVM sensitivity study. Run from repository root:

python -m experiments.part2_post_midterm.w10_svm

Only the original training partition is read. C is the sole varied setting.
This uses sklearn/libsvm; it is not a from-scratch SVM implementation.
"""
import csv
import hashlib
import inspect
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, _base
from threadpoolctl import threadpool_limits

from src.data import ACTIVITY_LABELS, RAW_DIR, load_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/w10_svm"
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
LABELS = np.arange(1, 7)


def metrics(y, pred):
    return {"macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist()}


def binary_checks():
    """Check dual decision reconstruction, box constraints, and primal/dual gap."""
    X = np.array([[-2., -1.], [-1., -1.], [-0.2, 0.4],
                  [0.3, -0.2], [1., 1.], [2., 1.]])
    y = np.array([-1, -1, 1, -1, 1, 1])
    model = SVC(C=0.7, kernel="linear", tol=1e-8).fit(X, y)
    signed_alpha = model.dual_coef_[0]
    alpha = np.abs(signed_alpha)
    w = signed_alpha @ model.support_vectors_
    decision = X @ w + model.intercept_[0]
    np.testing.assert_allclose(decision, model.decision_function(X), atol=1e-10)
    assert abs(signed_alpha.sum()) < 1e-8
    assert np.all(alpha >= 0) and np.all(alpha <= model.C + 1e-8)
    hinge = np.maximum(0, 1-y*decision)
    primal = 0.5*np.dot(w, w) + model.C*hinge.sum()
    dual = alpha.sum() - 0.5*np.dot(w, w)
    assert -1e-7 <= primal-dual < 1e-6
    return {"primal": float(primal), "dual": float(dual),
            "duality_gap": float(primal-dual), "box_and_balance_constraints_pass": True,
            "decision_reconstruction_pass": True}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = binary_checks()
    X, y, subjects = load_data("train")
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, subjects))
    assert not set(subjects[fit]) & set(subjects[val])
    assert set(y[fit]) == set(LABELS) == set(y[val])
    # StandardScaler sees only fit rows; validation cannot influence statistics.
    scaler = StandardScaler().fit(X[fit])
    Xfit, Xval = scaler.transform(X[fit]), scaler.transform(X[val])
    assert int(scaler.n_samples_seen_) == len(fit)
    np.testing.assert_allclose(scaler.mean_, X[fit].mean(axis=0))
    audit = {"date": "2026-11-03", "split_seed": 42, "test_loaded": False,
             "fit_subjects": sorted(map(int, np.unique(subjects[fit]))),
             "validation_subjects": sorted(map(int, np.unique(subjects[val]))),
             "fit_windows": len(fit), "validation_windows": len(val), "features": X.shape[1],
             "C_values": C_VALUES, "selection_rule": "highest validation macro-F1; smaller C on exact ties",
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
             "model": "sklearn.svm.SVC", "kernel": "linear", "multiclass": "one-versus-one (15 classifiers)",
             "settings": {"tol": 1e-3, "cache_size_MB": 512, "probability": False, "class_weight": None,
                          "shrinking": True, "max_iter": -1, "decision_function_shape": "ovo"},
             "binary_checks": checks, "input_sha256": {}, "runs": []}
    for name in ("X_train.txt", "y_train.txt", "subject_train.txt"):
        path = RAW_DIR / "UCI HAR Dataset/train" / name
        audit["input_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit["sklearn_source"] = {}
    for name, obj in (("SVC", SVC), ("BaseSVC", _base.BaseSVC)):
        path = Path(inspect.getsourcefile(obj))
        audit["sklearn_source"][name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    predictions, rows = [], []
    for C in C_VALUES:
        model = SVC(C=C, kernel="linear", tol=1e-3, cache_size=512,
                    probability=False, decision_function_shape="ovo")
        start = perf_counter()
        model.fit(Xfit, y[fit])
        fit_seconds = perf_counter()-start
        assert model.fit_status_ == 0
        train_pred = model.predict(Xfit)
        start = perf_counter()
        pred = model.predict(Xval)
        predict_seconds = perf_counter()-start
        train_m, val_m = metrics(y[fit], train_pred), metrics(y[val], pred)
        # For linear SVC, the 15 pairwise scores equal Xw+b in scaled coordinates.
        manual = Xval @ model.coef_.T + model.intercept_
        score_error = float(np.max(np.abs(manual-model.decision_function(Xval))))
        assert score_error < 1e-7
        widths = 2/np.linalg.norm(model.coef_, axis=1)
        stored_bytes = sum(a.nbytes for a in (model.support_vectors_, model.dual_coef_, model.intercept_, model.support_))
        record = {"C": C, "train": train_m, "validation": val_m, "fit_seconds": fit_seconds,
                  "validation_predict_seconds": predict_seconds, "support_vectors": int(model.n_support_.sum()),
                  "support_vectors_by_class": model.n_support_.tolist(), "fit_status": int(model.fit_status_),
                  "iterations_by_pair": model.n_iter_.tolist(), "pairwise_full_margin_widths": widths.tolist(),
                  "decision_reconstruction_max_abs_error": score_error,
                  "core_model_array_bytes": stored_bytes}
        audit["runs"].append(record)
        rows.append({"C": C, "train_macro_f1": train_m["macro_f1"], "validation_macro_f1": val_m["macro_f1"],
                     "validation_accuracy": val_m["accuracy"], "support_vectors": record["support_vectors"],
                     "fit_seconds": fit_seconds, "validation_predict_seconds": predict_seconds})
        for i, label in zip(val, pred):
            predictions.append({"C": C, "source_row_1based": int(i+1), "subject": int(subjects[i]),
                                "true": int(y[i]), "predicted": int(label)})
        print(f"C={C:g}: train F1={train_m['macro_f1']:.6f}, val F1={val_m['macro_f1']:.6f}, SV={record['support_vectors']}", flush=True)
    best = max(audit["runs"], key=lambda r: (r["validation"]["macro_f1"], -r["C"]))
    audit["selected_C"] = best["C"]
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    for filename, data in (("metrics.csv", rows), ("predictions.csv", predictions)):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].semilogx(C_VALUES, [r["train"]["macro_f1"] for r in audit["runs"]], "o-", label="Fit")
    axes[0].semilogx(C_VALUES, [r["validation"]["macro_f1"] for r in audit["runs"]], "o-", label="Validation")
    axes[0].set(xlabel="C", ylabel="Macro-F1", title="Linear SVM: fixed subject split")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].semilogx(C_VALUES, [r["support_vectors"] for r in audit["runs"]], "o-")
    axes[1].set(xlabel="C", ylabel="Number of support vectors", title="Model support")
    axes[1].grid(alpha=.25)
    fig.savefig(OUT / "c_sensitivity.png", dpi=160); plt.close(fig)
    cm = np.array(best["validation"]["confusion_matrix"])
    offdiag = cm.copy(); np.fill_diagonal(offdiag, 0)
    a, b = np.unravel_index(offdiag.argmax(), offdiag.shape)
    lines = ["# W10 - Linear SVM on HAR", "", "Run: `python -m experiments.part2_post_midterm.w10_svm`", "",
        "Fit: 5,270 windows / 15 subjects. Validation: 2,082 windows / 6 disjoint subjects. Official test files are not read.", "",
        "All 561 features are standardized using fit-only statistics. SVC uses the linear kernel and 15 one-versus-one binary classifiers. Only C varies. No probability calibration or stochastic training is enabled, so repeated seed runs on this fixed split are omitted.", "",
        "| C | Fit Macro-F1 | Validation Macro-F1 | Validation accuracy | SV count | Fit seconds |", "|---:|---:|---:|---:|---:|---:|"]
    for r in audit["runs"]:
        lines.append(f"| {r['C']:g} | {r['train']['macro_f1']:.6f} | {r['validation']['macro_f1']:.6f} | {r['validation']['accuracy']:.6f} | {r['support_vectors']} | {r['fit_seconds']:.3f} |")
    lines += ["", "![C sensitivity](c_sensitivity.png)", "", f"Selected C: {best['C']:g}, by validation Macro-F1 (smaller C breaks exact ties). This is a tuning result on one holdout, not an unbiased final-test score.", "",
        "## Selected-model confusion matrix", "", "Rows = true, columns = predicted; order: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.", "", "```text", np.array2string(cm), "```", "",
        f"Largest directed confusion: {ACTIVITY_LABELS[int(a+1)]} -> {ACTIVITY_LABELS[int(b+1)]}: {cm[a,b]} windows.", "",
        "## Five error examples", "", "First five errors in original validation row order, not cherry-picked.", "",
        "| Source row | Subject | True | Predicted |", "|---:|---:|---|---|"]
    errors = [r for r in predictions if r["C"] == best["C"] and r["true"] != r["predicted"]][:5]
    for r in errors:
        lines.append(f"| {r['source_row_1based']} | {r['subject']} | {ACTIVITY_LABELS[r['true']]} | {ACTIVITY_LABELS[r['predicted']]} |")
    lines += ["", "## Checks and interpretation", "",
        "A separate binary example verifies dual-score reconstruction, dual box/equality constraints, and a small primal-dual gap. For every HAR run, the 15 pairwise scores match explicit Xw+b; all solvers converge. These checks verify the experiment's interpretation of sklearn outputs, not execution parity with the upstream CVXOPT implementation.", "",
        "Increasing C penalizes hinge loss more heavily relative to the norm penalty. It need not improve validation accuracy. The widths in audit.json are the 15 pairwise canonical widths in standardized coordinates, not a single multiclass margin. Support-vector counts are unique across pairs. Model byte counts include selected fitted arrays only, not serialization or process memory.", "",
        "Quantitative comparison is restricted to this subject split. Correlated overlapping windows are not independent trials; no significance claim is made. Fit times are single-run wall-clock measurements with numerical thread pools limited to one. Kernel/RBF comparison is reserved for W11."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
