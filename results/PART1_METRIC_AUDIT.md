# Part I metric and validation audit -- 2026-10-14

This is a dated, AI-assisted correction of the existing working tree, not a new independent first attempt. Original checkpoint tags and Git history have not been changed.

## Corrected historical test results

HAR labels are 1--6. The old metric allocated seven classes, including unused class 0, whose zero F1 reduced the average by a factor of 6/7. The default now averages observed labels; experiments explicitly request labels 1--6. `n_classes` remains available for an explicitly zero-based class universe.

| Model | Original recorded F1 | Original recorded accuracy | 14 Oct rerun F1 | 14 Oct rerun accuracy |
|---|---:|---:|---:|---:|
| Majority | 0.0440 | 0.1822 | 0.051378 | 0.182219 |
| Stratified random, seed 42 | 0.1387 | 0.1639 | 0.161759 | 0.163895 |
| OvR Perceptron, lr 0.01, 100 epochs, seed 42 | 0.8143 | 0.9494 | 0.954450 | 0.953851 |

The Perceptron rerun is NOT just a rescaling of the old row: the old F1 would rescale to about 0.9500, and its old accuracy is also different. No stored original predictions exist to reconcile this discrepancy. The current source with the fixed historical configuration was rerun, and the new values are explicitly labelled as rerun results. Runtime is machine-dependent (7.09 s here versus the old recorded 12.50 s).

The original learning-rate script repeatedly evaluated final test data. That exposure cannot be undone. The corrected historical test results are descriptive results on an already-exposed test partition, not an unbiased fresh assessment. No new test configuration was searched, and GaussianNB has not been evaluated on final test data in this audit.

## Comparable validation results

The original official train/test population is unchanged. A new internal split uses `GroupShuffleSplit(test_size=0.25, random_state=42)` on the 21 original training subjects only. Six subjects are held out (rounding 25% of 21 upward): 1, 3, 8, 15, 25, 27. The other 15 subjects provide 5,270 fit windows; validation has 2,082 windows. Scaling parameters are fit on the fit rows only. This split is an audit-time validation split, not historical evidence of how the original models were chosen.

| Model | Validation Macro-F1 | Validation accuracy |
|---|---:|---:|
| Majority | 0.052020 | 0.184918 |
| Stratified random | 0.164538 | 0.167627 |
| OvR Perceptron | 0.949570 | 0.954371 |
| GaussianNB | 0.722378 | 0.736311 |

All five Perceptron learning rates (0.001, 0.005, 0.01, 0.05, 0.1) produce identical validation scores. Starting from zero, a fixed positive learning rate scales all weights and biases together in the exact-arithmetic update rule; the prediction signs and OvR ranking need not change. This experiment shows no learning-rate benefit. The first tied candidate, 0.001, is saved deterministically, while the historical test rerun keeps lr 0.01. A single held-out split and seed do not measure population-wide uncertainty, and the validation score used for selecting a candidate is not an independent post-selection test score.

## GaussianNB correction and diagnostics

The required smoothing is `var_smoothing * X.var(axis=0).max()` using all fit rows, not `var_smoothing * max(per_class_variance)`. A regression test with widely separated means and substantial smoothing detects the old error even though the original contract test passed at its tolerance.

On HAR validation, prediction agreement with installed scikit-learn is 100%; maximum absolute normalised log-posterior difference is approximately 5.59e-9. The largest directed confusion is STANDING to SITTING: 264 windows. Mean maximum posterior is 0.997088, while accuracy is 0.736311 and multiclass log loss is 7.150856 (computed with the installed sklearn implementation's probability clipping). These figures indicate overconfidence; normalising probabilities does not calibrate them. Correlated features are a plausible contributor, not a causal finding established by this experiment.

## Reproduce

From the repository root, with `requirements.txt` installed and the existing HAR files under `data/raw/`:

```text
python experiments/part1_pre_midterm/w05_baseline.py
python experiments/part1_pre_midterm/w05_perceptron.py
python experiments/part1_pre_midterm/w06_naive_bayes.py
python -m pytest -q tests/test_metrics.py tests/test_naive_bayes.py tests/test_evaluation.py
```

To reproduce the explicitly labelled historical test audit only:

```text
python experiments/part1_pre_midterm/w05_baseline.py --evaluate-test
python experiments/part1_pre_midterm/w05_perceptron.py --evaluate-test
```

`metrics.csv` contains historical test reruns; `validation_metrics.csv` contains internal validation. They must not be merged into one unlabelled comparison. Audit JSON files record labels, subject IDs, confusion matrices, settings and unrounded metrics. `audit_environment.json` records the actual runtime versions. CSV updates replace a model row without deleting other models or accumulating duplicate rows.

Verification on 14 Oct: 30 tests passed and 15 optional tests were skipped at checkpoint W07. On this Windows sandbox, pytest's default temporary directory was unavailable. The successful full-suite run used `CO3117_WEEK=W07` and `python -m pytest -q -rfs tests --basetemp=results/temp/pytest-metric-audit` after creating the parent directory `results/temp`.

## Remaining Part I work

Follow-up work on 14 Oct added the attributed Tree/GA adaptations, MLP capacity experiments, BN/TAN checks and W07 handwritten worked-review scan/corrections. Scan evidence fcfbdc8 precedes practice/corrections revision f44ef9e; see PROGRESS.md and `PART1_PRACTICE_RESULTS.md`. Independent ownership/reproduction is not established by automated checks. The Part I report and two-A4 sheet remain deferred; the final package tag has not been created or pushed.
