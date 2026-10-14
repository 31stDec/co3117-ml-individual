"""Perceptron: validation-only LR sweep, or fixed historical test audit.

Default holds out subjects from original train. --evaluate-test reruns ONLY
lr=0.01, epochs=100, seed=42 to correct the previously exposed test result.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import ACTIVITY_LABELS
from src.from_scratch.perceptron import MultiClassPerceptron
from src.metrics import accuracy, confusion_matrix, macro_f1, print_classification_report
from experiments.part1_pre_midterm.common import (
    LABELS, ROOT, evaluation_data, save_metric, standardise,
)

RANDOM_SEED = 42


def run_experiment(learning_rate=0.01, n_iters=100, *, evaluate_test=False, data=None):
    if data is None:
        data = evaluation_data(evaluate_test, RANDOM_SEED)
    X, y, Xe, ye, metadata = data
    Xs, Xes = standardise(X, Xe)
    clf = MultiClassPerceptron(learning_rate=learning_rate, n_iters=n_iters,
                             random_seed=RANDOM_SEED)
    start = time.perf_counter()
    clf.fit(Xs, y)
    runtime = time.perf_counter() - start
    train_pred, pred = clf.predict(Xs), clf.predict(Xes)
    return {
        "lr": learning_rate, "n_iters": n_iters, "train_time_s": runtime,
        "train_f1": macro_f1(y, train_pred, labels=LABELS),
        "evaluation_f1": macro_f1(ye, pred, labels=LABELS),
        "train_acc": accuracy(y, train_pred), "evaluation_acc": accuracy(ye, pred),
        "confusion_matrix": confusion_matrix(ye, pred, labels=LABELS).tolist(),
        "metadata": metadata, "y_evaluation": ye, "predictions": pred,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    data = evaluation_data(args.evaluate_test, RANDOM_SEED)
    rates = [0.01] if args.evaluate_test else [0.001, 0.005, 0.01, 0.05, 0.1]
    records = []
    for rate in rates:
        record = run_experiment(rate, data=data)
        print(f"lr={rate:g}: {record['metadata']['split']} Macro-F1="
              f"{record['evaluation_f1']:.6f}, accuracy={record['evaluation_acc']:.6f}", flush=True)
        records.append(record)
    best = max(records, key=lambda record: record["evaluation_f1"])
    print_classification_report(best["y_evaluation"], best["predictions"],
                                [ACTIVITY_LABELS[c] for c in LABELS], labels=LABELS)
    split = best["metadata"]["split"]
    note = ("Historical fixed configuration; test previously exposed; corrected six-class F1"
            if args.evaluate_test else "LR chosen on subject-held-out validation only")
    save_metric({
        "model": "perceptron_ovr", "chapter": 4, "depth": "Depth-A", "week": "W05",
        "macro_f1": f"{best['evaluation_f1']:.6f}",
        "accuracy": f"{best['evaluation_acc']:.6f}",
        "train_time_s": f"{best['train_time_s']:.2f}",
        "notes": f"{note}; lr={best['lr']}; epochs=100; seed=42",
    }, split)
    serializable = [{k: v for k, v in r.items() if k not in ("y_evaluation", "predictions")}
                    for r in records]
    (ROOT / "results" / f"perceptron_{split}_audit.json").write_text(
        json.dumps({"labels": LABELS.tolist(), "runs": serializable}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
