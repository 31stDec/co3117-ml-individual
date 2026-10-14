# Part I experiments

Run from the repository root with the HAR dataset already available locally:

```text
python experiments/part1_pre_midterm/w05_baseline.py
python experiments/part1_pre_midterm/w05_perceptron.py
python experiments/part1_pre_midterm/w06_naive_bayes.py
```

All three commands now default to the same subject-held-out validation split within the original training population (seed 42). They do not open final test files. `w05_perceptron.py` sweeps learning rates on validation only; `w06_naive_bayes.py` compares with the installed sklearn reference and records probability/error diagnostics.

Validation scores are saved to `results/validation_metrics.csv`. JSON audit files include the actual subject IDs, settings, six-class confusion matrices and unrounded metrics.

The `--evaluate-test` option on the two W05 scripts explicitly reruns historical fixed configurations on the already-exposed test partition and updates `results/metrics.csv`. It must not be used for parameter search. No test evaluation is provided by the W06 script.

See `results/PART1_METRIC_AUDIT.md` for the dated correction, original values, runtime versions and reproducibility limits.

## Tree, MLP, GA and BN/TAN practice

```text
python experiments/part1_pre_midterm/w05_tree.py
python experiments/part1_pre_midterm/w05_mlp.py
python experiments/part1_pre_midterm/w07_ga.py
python experiments/part1_pre_midterm/w07_bn.py
python experiments/part1_pre_midterm/summarize_practice.py
```

Run in this order to create the saved results and three figures. Defaults are fixed in `docs/experiments/PART1_PLAN.md`. Tree varies depth/minimum leaf size and benchmarks sklearn pruning; MLP varies width 32/64 across seeds 42/0/1 using the existing NumPy core; GA varies zero/one elite on a nested subject split with the same three seeds. BN/TAN checks are illustrative probability enumeration plus fit-only HAR feature-dependence evidence.

These commands never load the official final-test subset. GA feature ranking and fitness use only an inner split of outer-fit subjects. Predictions for the chosen mask are evaluated on outer validation after search, and no variant is selected using final-test scores.

Detailed conclusions and limitations are in `results/PART1_PRACTICE_RESULTS.md`; source attribution and exact function/line mapping are in `docs/reference/DISSECTION.md`. Current report and handwritten A4 submission status is recorded in `SUBMISSION_PART1.md`.

The recorded environment uses Python 3.13.13. For its tested dependencies, use `requirements-part1-lock.txt`; the broader `requirements.txt` remains the course setup file. Numerical kernels in the new paired experiments use one BLAS thread.

Tests: set `CO3117_WEEK=W07`, then `python -m pytest -q -rfs tests`. If this Windows sandbox cannot create pytest's default temporary directory, create `results/temp` first and add `--basetemp=results/temp/pytest-practice-audit`.
