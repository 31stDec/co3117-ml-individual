import numpy as np
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.reference_adapters.decision_tree import ContinuousTree
from src.reference_adapters.genetic_algorithm import FeatureGA
from src.nn_training import MLPClassifier


def test_tree_midpoint_and_predictions_match_known_split():
    X, y = np.array([[0.], [1.], [3.], [4.]]), np.array([1, 1, 6, 6])
    own = ContinuousTree().fit(X, y)
    reference = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X, y)
    assert own.root_.threshold == 2.0
    assert np.array_equal(own.predict(X), reference.predict(X))
    assert own.depth_ == 1 and own.n_leaves_ == 2


def test_tree_depth_zero_and_child_size_prevent_splits():
    X, y = np.arange(7).reshape(-1, 1), np.array([1, 1, 1, 6, 6, 6, 6])
    for model in [ContinuousTree(max_depth=0), ContinuousTree(min_samples_leaf=4)]:
        model.fit(X, y)
        assert model.n_leaves_ == 1 and model.depth_ == 0
        assert np.all(model.predict(X) == 6)


def test_tree_ignores_constant_features_and_limits_leaf_sizes():
    X = np.column_stack([np.ones(12), np.arange(12)])
    y = np.array([1] * 6 + [6] * 6)
    model = ContinuousTree(min_samples_leaf=3).fit(X, y)
    assert model.root_.feature == 1
    assert np.array_equal(model.predict(X), y)
    assert np.isclose(model.feature_importances_.sum(), 1)


def test_ga_handles_negative_fitness_and_preserves_elite():
    def fitness(mask):
        assert mask.any()
        return -float(mask.sum())
    first = FeatureGA(6, mutation_rate=1.0, generations=8, seed=3).run(fitness)
    second = FeatureGA(6, mutation_rate=1.0, generations=8, seed=3).run(fitness)
    scores = [r["generation_best"] for r in first.history_]
    assert np.all(np.diff(scores) >= 0)
    assert np.array_equal(first.best_mask_, second.best_mask_)
    assert first.history_ == second.history_
    assert all(0 <= r["diversity"] <= 1 for r in first.history_)


def test_ga_elitism_comparison_starts_with_same_population():
    a = FeatureGA(6, seed=7, elite_count=0).run(lambda mask: float(mask.sum()))
    b = FeatureGA(6, seed=7, elite_count=1).run(lambda mask: float(mask.sum()))
    assert np.array_equal(a.initial_population_, b.initial_population_)
    assert np.isclose(a.best_fitness_, a.best_mask_.sum())


def test_mlp_training_reduces_loss_and_learns_separated_clusters():
    rng = np.random.default_rng(5)
    centers = np.array([[-3., -3.], [3., -3.], [0., 3.]])
    y = np.repeat([1, 3, 6], 30)
    X = np.repeat(centers, 30, axis=0) + rng.normal(scale=0.2, size=(90, 2))
    with threadpool_limits(limits=1):
        model = MLPClassifier(hidden_width=8, epochs=35, batch_size=15).fit(X, y, X, y)
    assert model.history_[-1]["train_loss"] < model.history_[0]["train_loss"]
    assert np.mean(model.predict(X) == y) > 0.98
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
