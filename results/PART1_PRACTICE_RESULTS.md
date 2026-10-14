# Part I practice results - 2026-10-14

All results below use the same outer train-subject validation split. No new final-test data was evaluated. These are dated Codex-assisted experiments; they do not replace historical HW2/HW3 scans or establish independent authorship.

## Tree stopping and pruning

| Configuration | Train F1 | Validation F1 | Validation accuracy | Depth | Leaves | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| tree_adapted_unlimited | 1.0000 | 0.8624 | 0.8713 | 14 | 100 | 3.41 |
| tree_adapted_depth6 | 0.9529 | 0.8719 | 0.8833 | 6 | 38 | 2.44 |
| tree_adapted_leaf20 | 0.9569 | 0.8546 | 0.8646 | 11 | 54 | 2.66 |
| tree_sklearn_unpruned | 1.0000 | 0.8815 | 0.8881 | 14 | 100 | 1.24 |
| tree_sklearn_pruned | 0.9898 | 0.8706 | 0.8775 | 12 | 65 | 1.25 |

Depth six reduces leaves from 100 to 38 and improves validation F1 from 0.8624 to 0.8719 in the adapted tree. Minimum leaf size 20 reduces complexity but lowers validation F1 to 0.8546. sklearn post-pruning reduces leaves from 100 to 65 but lowers validation F1 from 0.8815 to 0.8706. Regularization therefore does not uniformly improve this holdout. The implementations use different threshold/tie behavior and are not claimed to predict identically on HAR.

## MLP capacity

| Width | Seed | Train F1 | Validation F1 | Validation accuracy | Fit time (s) | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 42 | 0.9964 | 0.9522 | 0.9573 | 1.22 | 18182 |
| 64 | 42 | 0.9974 | 0.9514 | 0.9558 | 2.13 | 36358 |
| 32 | 0 | 0.9926 | 0.9563 | 0.9597 | 1.26 | 18182 |
| 64 | 0 | 0.9964 | 0.9539 | 0.9582 | 2.11 | 36358 |
| 32 | 1 | 0.9945 | 0.9400 | 0.9448 | 1.24 | 18182 |
| 64 | 1 | 0.9954 | 0.9460 | 0.9505 | 2.07 | 36358 |

All runs use tanh, one hidden layer, existing NumPy forward/backward, mini-batch SGD, lr 0.05, 60 epochs and batch size 128. No early stopping or validation-selected epoch is used.

- Width 32: validation F1 mean 0.949486, sample standard deviation 0.008478; mean fit time 1.24 s.
- Width 64: validation F1 mean 0.950444, sample standard deviation 0.004050; mean fit time 2.10 s.

The mean width advantage is about 0.001, smaller than the observed seed variability; 64 units are not consistently better (two seed pairs favor 32). More capacity increases train performance and runtime, without a demonstrated robust generalization gain. Seed variation is not population uncertainty. Hardware timings use one BLAS thread and are not directly comparable to the old HW3 Adam timings.

## GA feature selection and elitism

Candidates are the top 12 ANOVA features ranked on 3,911 windows from 11 inner-fit subjects. Fitness uses 1,359 windows from four separate inner-validation subjects. The chosen mask is refit on 5,270 outer-fit windows and evaluated on the untouched-by-search outer validation. All 12 candidates with GNB give outer validation F1 0.740683.

| Seed | Elites | Best inner fitness | Inner F1 | Features | Outer validation F1 | Unique fitness evaluations | Final diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 0.674613 | 0.686279 | 7 | 0.731027 | 106 | 0.1931 |
| 42 | 1 | 0.674613 | 0.686279 | 7 | 0.731027 | 91 | 0.2174 |
| 0 | 0 | 0.674613 | 0.686279 | 7 | 0.731027 | 124 | 0.2694 |
| 0 | 1 | 0.674613 | 0.686279 | 7 | 0.731027 | 137 | 0.2882 |
| 1 | 0 | 0.674613 | 0.686279 | 7 | 0.731027 | 112 | 0.3090 |
| 1 | 1 | 0.672145 | 0.685478 | 8 | 0.740619 | 85 | 0.2271 |

Population size 16; 12 evaluated generations; tournament size three; single-point crossover; bit mutation probability 1/12; fitness = inner F1 - 0.02 * count/12. Paired variants start with identical populations. Elites cannot mutate. Both variants return an all-time best, not necessarily the last generation's best.

Elitism prevents a decline in current-generation best fitness but does not ensure a better discovered optimum. Seed 1 with one elite finds a lower inner-fitness 8-feature mask with higher outer F1 than the 7-feature mask found in other runs. This illustrates that inner fitness ranking does not perfectly transfer to outer subjects. GA gives feature compression rather than a demonstrated accuracy gain over the all-12 baseline. Three seeds and a small candidate pool do not prove global optimality.

## BN/TAN checks and suitability

The specified five-binary-variable worked network normalizes to one; enumeration agrees with all four d-separation examples and the 10-parameter count. This is an illustrative mathematical check, not a second predictive benchmark dataset.

In HAR fit rows, `tBodyAccMag-mean()` and `tBodyAccMag-energy()` have within-class Pearson correlations 0.868--0.988. This is empirical evidence against strict conditional independence for this pair, not proof that a particular fitted TAN will improve classification. TAN could capture one feature dependency per node, but a continuous TAN requires conditional density estimation; discretization would require fit-only binning and add information loss. No fitted TAN performance is claimed.

## Artifacts and reproduction

Settings and hypotheses were recorded before running in `docs/experiments/PART1_PLAN.md`. Reference attribution, exact saved source hashes and equation-to-function mapping are in `docs/reference/DISSECTION.md` and `sources.json`. Full per-run confusion matrices, feature names, losses, diversity and runtime are in the audit JSON files. Curves are in `results/figures/`.

```text
python experiments/part1_pre_midterm/w05_tree.py
python experiments/part1_pre_midterm/w05_mlp.py
python experiments/part1_pre_midterm/w07_ga.py
python experiments/part1_pre_midterm/w07_bn.py
python experiments/part1_pre_midterm/summarize_practice.py
```

No independent pre-reference W07 code attempt or closed-book reproduction is claimed. Validation reuse, the historical test exposure documented in `PART1_METRIC_AUDIT.md`, and the small seed count limit conclusions. Current report and handwritten A4 submission status is recorded in `SUBMISSION_PART1.md`.
