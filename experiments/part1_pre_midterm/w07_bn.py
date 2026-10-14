"""BN worked checks and HAR suitability evidence; no second benchmark dataset."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.part1_pre_midterm.common import evaluation_data
from experiments.part1_pre_midterm.practice_common import feature_names, write_json


def illustrative_joint():
    # Five binary variables A -> C <- B, C -> D, B -> E.
    # Specified CPTs are a worked algebra example, not a replacement HAR task.
    joint = np.zeros((2, 2, 2, 2, 2))
    c_probability = np.array([[0.05, 0.65], [0.70, 0.95]])
    for a, b, c, d, e in itertools.product(range(2), repeat=5):
        probabilities = [0.4, 0.3, c_probability[a, b], [0.1, 0.9][c], [0.2, 0.8][b]]
        joint[a, b, c, d, e] = np.prod([p if value else 1 - p
                                      for value, p in zip([a, b, c, d, e], probabilities)])
    return joint


def conditional_table(joint, pair, evidence):
    table = np.zeros((2, 2))
    for states in itertools.product(range(2), repeat=5):
        if all(states[index] == value for index, value in evidence.items()):
            table[states[pair[0]], states[pair[1]]] += joint[states]
    return table / table.sum()


def main():
    joint = illustrative_joint()
    assert np.isclose(joint.sum(), 1.0)
    checks = []
    for description, pair, evidence, expected in [
        ("A independent B without evidence", (0, 1), {}, True),
        ("A independent B given C=1", (0, 1), {2: 1}, False),
        ("A independent B given D=1", (0, 1), {3: 1}, False),
        ("A independent D given C=1", (0, 3), {2: 1}, True),
    ]:
        table = conditional_table(joint, pair, evidence)
        factored = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :]
        independent = bool(np.allclose(table, factored, atol=1e-10))
        assert independent == expected
        checks.append({"description": description, "independent_for_specified_CPTs": independent,
                       "conditional_joint": table.tolist(),
                       "max_abs_factorization_residual": float(np.max(np.abs(table - factored)))})
    X, y, _, _, metadata = evaluation_data()
    names = feature_names()
    pair_names = ["tBodyAccMag-mean()", "tBodyAccMag-energy()"]
    indices = [names.index(name) for name in pair_names]
    correlations = [{"class_label": int(label), "n_fit_windows": int((y == label).sum()),
                     "pearson_correlation": float(np.corrcoef(X[y == label][:, indices].T)[0, 1])}
                    for label in np.unique(y)]
    result = {"metadata": metadata, "illustration": "specified five-binary-variable CPTs; not a predictive benchmark",
              "independent_parameter_count": 10, "enumeration_checks": checks,
              "HAR_feature_pair": pair_names, "feature_indices_zero_based": indices,
              "within_class_fit_correlations": correlations,
              "limitation": "TAN is studied structurally; no fitted continuous TAN classifier or TAN performance claim."}
    write_json("bn_tan_checks.json", result)
    print("BN enumeration: all four worked checks passed; parameter count=10")
    print("HAR within-class correlations:", correlations)


if __name__ == "__main__":
    main()
