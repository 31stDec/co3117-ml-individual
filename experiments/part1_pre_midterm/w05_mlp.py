"""Paired 32/64-unit NumPy MLP capacity experiment; validation only."""
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nn_training import MLPClassifier
from experiments.part1_pre_midterm.common import evaluation_data, standardise, LABELS
from experiments.part1_pre_midterm.practice_common import scores, save_run_metric, write_json


def main():
    X, y, Xv, yv, metadata = evaluation_data()
    X, Xv = standardise(X, Xv)
    runs = []
    with threadpool_limits(limits=1):
        for seed in [42, 0, 1]:
            for width in [32, 64]:
                model = MLPClassifier(hidden_width=width, seed=seed)
                start = time.perf_counter()
                model.fit(X, y, Xv, yv)
                runtime = time.perf_counter() - start
                record = {"model": f"mlp_width{width}_seed{seed}", "hidden_width": width,
                          "seed": seed, "epochs": 60, "batch_size": 128, "lr": 0.05,
                          "optimizer": "mini-batch SGD", "hidden_activation": "tanh",
                          "train_time_s": runtime,
                          "parameter_count": int(sum(p.size for p in model.params_.values())),
                          "history": model.history_,
                          **scores(y, model.predict(X), yv, model.predict(Xv))}
                print(f"MLP width={width}, seed={seed}: train F1={record['train_macro_f1']:.4f}, "
                      f"validation F1={record['validation_macro_f1']:.4f}, time={runtime:.2f}s", flush=True)
                runs.append(record)
                save_run_metric(record, 3, "Depth-A", "W05",
                                f"Subject-held-out validation; width={width}; seed={seed}; SGD lr=0.05; epochs=60")
    summaries = []
    for width in [32, 64]:
        group = [r for r in runs if r["hidden_width"] == width]
        summaries.append({"width": width, "n_seeds": len(group),
                          "mean_validation_f1": float(np.mean([r["validation_macro_f1"] for r in group])),
                          "std_validation_f1": float(np.std([r["validation_macro_f1"] for r in group], ddof=1)),
                          "mean_train_time_s": float(np.mean([r["train_time_s"] for r in group]))})
    write_json("mlp_validation_audit.json", {"metadata": metadata, "labels": LABELS.tolist(),
                                            "runs": runs, "summary": summaries})


if __name__ == "__main__":
    main()
