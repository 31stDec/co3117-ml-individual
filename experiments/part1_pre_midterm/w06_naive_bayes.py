"""GaussianNB on subject-held-out validation; no final test access or tuning."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB as Reference
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.from_scratch.naive_bayes import GaussianNB
from src.metrics import accuracy, confusion_matrix, macro_f1
from experiments.part1_pre_midterm.common import LABELS, ROOT, evaluation_data, save_metric


def main():
    X, y, Xv, yv, metadata = evaluation_data()
    start = time.perf_counter()
    own = GaussianNB(var_smoothing=1e-9).fit(X, y)
    runtime = time.perf_counter() - start
    ref = Reference(var_smoothing=1e-9).fit(X, y)
    logp, ref_logp = own.predict_log_proba(Xv), ref.predict_log_proba(Xv)
    pred = own.predict(Xv)
    comparison = {
        "max_abs_log_posterior_difference": float(np.max(np.abs(logp - ref_logp))),
        "prediction_agreement": float(np.mean(pred == ref.predict(Xv))),
        "epsilon": float(own.epsilon_),
    }
    assert np.allclose(logp, ref_logp, atol=1e-8, rtol=1e-8)
    assert comparison["prediction_agreement"] == 1.0
    score, acc = macro_f1(yv, pred, labels=LABELS), accuracy(yv, pred)
    cm = confusion_matrix(yv, pred, labels=LABELS)
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    error = np.unravel_index(np.argmax(off_diagonal), cm.shape)
    result = {
        "labels": LABELS.tolist(), "metadata": metadata,
        "macro_f1": score, "accuracy": acc, "train_time_s": runtime,
        "confusion_matrix": cm.tolist(), "reference_comparison": comparison,
        "probability_diagnostics": {
            "log_loss": float(log_loss(yv, np.exp(logp), labels=LABELS)),
            "mean_max_probability": float(np.exp(logp).max(axis=1).mean()),
        },
        "largest_directed_confusion": {
            "true_label": int(LABELS[error[0]]), "predicted_label": int(LABELS[error[1]]),
            "count": int(cm[error]),
        },
    }
    save_metric({"model": "gaussian_nb", "chapter": 6, "depth": "Depth-A", "week": "W06",
                 "macro_f1": f"{score:.6f}", "accuracy": f"{acc:.6f}",
                 "train_time_s": f"{runtime:.4f}",
                 "notes": "Subject-held-out validation; smoothing=1e-9; sklearn agreement=100%"}, "validation")
    (ROOT / "results" / "naive_bayes_validation_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
