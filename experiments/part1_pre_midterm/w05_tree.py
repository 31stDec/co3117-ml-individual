"""Depth-B tree adaptation and stopping/pruning audit on train-subject validation."""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reference_adapters.decision_tree import ContinuousTree
from src.from_scratch.tree_split import information_gain
from experiments.part1_pre_midterm.common import evaluation_data, LABELS
from experiments.part1_pre_midterm.practice_common import scores, save_run_metric, write_json, feature_names


def main():
    X, y, Xv, yv, metadata = evaluation_data()
    names = feature_names()
    settings = [
        ("tree_adapted_unlimited", ContinuousTree(), "adapted", None, 1, 0),
        ("tree_adapted_depth6", ContinuousTree(max_depth=6), "adapted", 6, 1, 0),
        ("tree_adapted_leaf20", ContinuousTree(min_samples_leaf=20), "adapted", None, 20, 0),
        ("tree_sklearn_unpruned", DecisionTreeClassifier(criterion="entropy", random_state=42), "sklearn", None, 1, 0),
        ("tree_sklearn_pruned", DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.002, random_state=42), "sklearn", None, 1, 0.002),
    ]
    runs = []
    with threadpool_limits(limits=1):
        for name, model, implementation, max_depth, min_leaf, alpha in settings:
            start = time.perf_counter()
            model.fit(X, y)
            runtime = time.perf_counter() - start
            record = {"model": name, "implementation": implementation, "train_time_s": runtime,
                      "max_depth_setting": max_depth, "min_samples_leaf": min_leaf, "ccp_alpha": alpha,
                      **scores(y, model.predict(X), yv, model.predict(Xv))}
            if implementation == "adapted":
                record.update({"depth": model.depth_, "leaves": model.n_leaves_})
                root = model.root_
                if root.feature is not None:
                    record["root_split"] = {"feature_index_zero_based": root.feature,
                                             "feature_name": names[root.feature], "threshold": root.threshold,
                                             "information_gain": information_gain(y, X[:, root.feature] <= root.threshold)}
            else:
                record.update({"depth": int(model.get_depth()), "leaves": int(model.get_n_leaves())})
            top = np.argsort(-model.feature_importances_, kind="stable")[:3]
            record["top_features"] = [{"index_zero_based": int(i), "name": names[i],
                                       "importance": float(model.feature_importances_[i])} for i in top]
            print(f"{name}: train F1={record['train_macro_f1']:.4f}, validation F1="
                  f"{record['validation_macro_f1']:.4f}, depth={record['depth']}, leaves={record['leaves']}", flush=True)
            runs.append(record)
            save_run_metric(record, 2, "Depth-B" if implementation == "adapted" else "Depth-C", "W05",
                            f"Subject-held-out validation; depth={max_depth}; min_leaf={min_leaf}; ccp_alpha={alpha}")
    write_json("tree_validation_audit.json", {"metadata": metadata, "labels": LABELS.tolist(), "runs": runs})


if __name__ == "__main__":
    main()
