"""Nested subject-split GA feature selection; compare elitism across three seeds."""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import load_data
from src.from_scratch.naive_bayes import GaussianNB
from src.metrics import macro_f1
from src.reference_adapters.genetic_algorithm import FeatureGA
from experiments.part1_pre_midterm.common import evaluation_data, LABELS
from experiments.part1_pre_midterm.practice_common import scores, save_run_metric, write_json, feature_names


def main():
    X, y, Xv, yv, metadata = evaluation_data()
    Xraw, yraw, subjects = load_data("train")
    outer_fit_mask = np.isin(subjects, metadata["fit_subjects"])
    assert np.array_equal(X, Xraw[outer_fit_mask]) and np.array_equal(y, yraw[outer_fit_mask])
    outer_subjects = subjects[outer_fit_mask]
    inner_fit, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                               .split(X, y, outer_subjects))
    assert not set(outer_subjects[inner_fit]) & set(outer_subjects[inner_val])
    ranking, _ = f_classif(X[inner_fit], y[inner_fit])
    candidates = np.argsort(-np.nan_to_num(ranking, nan=-np.inf), kind="stable")[:12]
    names = feature_names()
    Xi, Xiv = X[inner_fit][:, candidates], X[inner_val][:, candidates]
    yi, yiv = y[inner_fit], y[inner_val]
    penalty = 0.02

    def fitness(mask):
        model = GaussianNB().fit(Xi[:, mask], yi)
        return macro_f1(yiv, model.predict(Xiv[:, mask]), labels=LABELS) - penalty * mask.sum() / len(mask)

    start = time.perf_counter()
    baseline_model = GaussianNB().fit(X[:, candidates], y)
    baseline_time = time.perf_counter() - start
    baseline = {"model": "ga_nb_all12", "selected_count": 12, "train_time_s": baseline_time,
                **scores(y, baseline_model.predict(X[:, candidates]), yv, baseline_model.predict(Xv[:, candidates]))}
    save_run_metric(baseline, 5, "baseline", "W07", "GA candidate baseline; 12 train-ranked features; outer validation")
    runs = []
    with threadpool_limits(limits=1):
        for seed in [42, 0, 1]:
            initial_population = None
            for elite_count in [0, 1]:
                search = FeatureGA(12, seed=seed, elite_count=elite_count)
                start = time.perf_counter()
                search.run(fitness)
                search_time = time.perf_counter() - start
                if initial_population is None:
                    initial_population = search.initial_population_
                else:
                    assert np.array_equal(initial_population, search.initial_population_)
                chosen = candidates[search.best_mask_]
                start = time.perf_counter()
                model = GaussianNB().fit(X[:, chosen], y)
                refit_time = time.perf_counter() - start
                record = {"model": f"ga_nb_elite{elite_count}_seed{seed}", "seed": seed,
                          "elite_count": elite_count, "population_size": 16, "generations": 12,
                          "mutation_rate": 1 / 12, "penalty": penalty,
                          "selected_count": int(len(chosen)), "mask": search.best_mask_.astype(int).tolist(),
                          "selected_indices_zero_based": chosen.tolist(),
                          "selected_names": [names[i] for i in chosen],
                          "best_inner_fitness": search.best_fitness_,
                          "best_inner_macro_f1": float(search.best_fitness_ + penalty * len(chosen) / 12),
                          "fitness_evaluations": len(search.cache_), "history": search.history_,
                          "search_time_s": search_time, "refit_time_s": refit_time,
                          "train_time_s": search_time + refit_time,
                          **scores(y, model.predict(X[:, chosen]), yv, model.predict(Xv[:, chosen]))}
                print(f"GA seed={seed}, elites={elite_count}: inner fitness={search.best_fitness_:.4f}, "
                      f"features={len(chosen)}, outer validation F1={record['validation_macro_f1']:.4f}", flush=True)
                runs.append(record)
                save_run_metric(record, 5, "Depth-B", "W07",
                                f"Nested subject-split fitness; seed={seed}; elites={elite_count}; features={len(chosen)}; outer validation")
    inner_metadata = {"fit_subjects": sorted(map(int, np.unique(outer_subjects[inner_fit]))),
                      "validation_subjects": sorted(map(int, np.unique(outer_subjects[inner_val]))),
                      "fit_samples": len(inner_fit), "validation_samples": len(inner_val)}
    write_json("ga_validation_audit.json", {"metadata": metadata, "inner_split": inner_metadata,
                                           "candidate_indices_zero_based": candidates.tolist(),
                                           "candidate_names": [names[i] for i in candidates],
                                           "labels": LABELS.tolist(), "all_candidates_baseline": baseline, "runs": runs})


if __name__ == "__main__":
    main()
