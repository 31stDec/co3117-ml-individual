# Attributed reference dissection and adaptation - 2026-10-14

The upstream files are preserved unchanged with their MIT license. `sources.json` records exact content SHA-256 values, source URLs and retrieval date. The adapted modules were produced with Codex assistance after reading these sources; assistance is recorded in AI_USE.md.

## Decision tree

Source: [ML-From-Scratch decision_tree.py](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py); snapshot `decision_tree.py` in this directory.

| Equation or algorithm step | Upstream locator | Adapted locator |
|---|---|---|
| Recursive threshold partition | `DecisionTree._build_tree`, lines 72--134 | `src/reference_adapters/decision_tree.py`, `_build`, lines 79--95 |
| Weighted entropy reduction | `ClassificationTree._calculate_information_gain`, lines 257--265 | `_entropy_counts`, lines 22--27; `_best_split`, lines 51--77 |
| Majority class at a terminal node | `_majority_vote`, lines 267--276 | `_build`: maximum label count at each node |
| Route a prediction down branches | `predict_value`, lines 137--160 | `predict`, lines 97--104 |

Information gain is H(parent) - n_left/n * H(left) - n_right/n * H(right). The upstream searches unique observed thresholds through repeated partitions; the adaptation sorts each continuous feature, uses cumulative class counts, and evaluates only midpoints between distinct adjacent values. This gives the same set of nonempty continuous partitions up to routing convention, with vectorized scoring. Both training and prediction use `x <= threshold` consistently.

The upstream depth gate permits splitting when current_depth <= max_depth; the adaptation stops when depth >= max_depth, so a requested depth of six is actually respected. A minimum leaf-size constraint rejects cuts with too few observations in either child. This is a substantive regularization change, not a formatting change. The controlled comparison varies only depth or minimum leaf size within the adapted implementation. An additional sklearn ccp_alpha comparison evaluates post-pruning. Different implementations can resolve equal-gain ties and floating-point thresholds differently, so exact HAR prediction agreement is not required; the known-split regression test checks the mathematical behavior.

## Genetic algorithm

Source: [ML-From-Scratch genetic_algorithm.py](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/genetic_algorithm.py); snapshot `genetic_algorithm.py` in this directory.

| Algorithm step | Upstream locator | Adapted locator |
|---|---|---|
| Initialize a population | `_initialize`, lines 25--31 | `src/reference_adapters/genetic_algorithm.py`, `_initialize`, lines 30--32 |
| Evaluate fitness | `_calculate_fitness`, lines 33--46 | `run`, lines 46--85, callback and cache |
| Parent selection | `run`, lines 81--89 | `_select`, lines 42--44 |
| Single-point crossover | `_crossover`, lines 59--65 | `_crossover`, lines 37--40 |
| Independent allele mutation | `_mutate`, lines 48--57 | `_mutate`, lines 34--35, bit flip |
| Generational replacement | `run`, lines 67--98 | `run`, lines 46--85, optional elite retention |

The original solves target-string search using alphabetical-distance fitness. The adaptation uses a binary mask over 12 training-ranked HAR features. Its fitness is inner validation Macro-F1 - 0.02 * selected_count/12 for GaussianNB. Tournament selection replaces fitness-proportional sampling: subtracting a feature penalty can produce zero or negative fitness, for which raw roulette probabilities are invalid. Empty masks are repaired by selecting one random feature. Repeated masks use cached fitness.

The adaptation optionally retains one elite unchanged and records both current-generation best and all-time best. Returning the all-time best avoids silently losing a good candidate in the no-elitism run. The ablation changes only elite_count (0 versus 1) with identical initial populations for each seed. Elitism mathematically prevents a decline in current-generation best fitness, but does not guarantee higher outer-validation F1 or a global optimum. Mean normalized pairwise Hamming distance measures diversity.

## MLP ownership boundary

The existing NumPy `src/from_scratch/mlp.py` supplies tanh forward propagation, stable softmax, mean cross-entropy and explicit backpropagation. Its existing contract tests check analytic gradients against numerical differences. `src/nn_training.py` adds a Codex-assisted mini-batch SGD loop; it does not use a high-level MLP estimator. The new experiment varies hidden width 32/64 under fixed optimizer, epochs, batch size, learning rate, preprocessing and paired shuffle seeds. It does not recreate or overwrite the historical sklearn/Adam HW3 results.

## Validation boundaries

All parameter comparisons use the same outer validation subjects and leave final test files unopened. GA fitness has an additional inner subject split; feature ranking uses inner fit rows only. Search never sees outer validation scores before selecting a mask. Outer validation is reused across families for descriptive comparison, so it is not an independent final test. The classifier fit and feature ranking are refit only on appropriate training rows, and no fitted model binaries or raw data are committed.
