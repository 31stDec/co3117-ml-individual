# W09 HMM: equations, own code, and numpy-ml

Reviewed 2026-10-27. Own implementation was committed as `49f5595` before this reference comparison. This ordering is a code-reference chronology, not a claim that the implementation was written without assistance.

## Pinned source

Repository: https://github.com/ddbourgin/numpy-ml

Commit: `b0359af5285fbf9699d64fd5ec059493228af03e`.

Inspected source: `numpy_ml/hmm/hmm.py`, class `MultinomialHMM`; helper `numpy_ml/utils/misc.py::logsumexp`. Exact snapshots, upstream license, and SHA-256 hashes are in [numpy_ml_hmm](numpy_ml_hmm/provenance.json). Line numbers below refer to these saved files, using one-based numbering.

## Equation-to-code trace

| Operation | Mathematics | Own `src/from_scratch/hmm.py` | Upstream `numpy_ml_hmm/hmm.py` |
|---|---|---|---|
| Forward initialization | alpha_1(i) = pi_i B_i(o_1) | `forward`, lines 51-56 (t=0) | `_forward`, lines 343-347 |
| Forward recursion | alpha_t(i) = B_i(o_t) sum_k alpha_(t-1)(k) A_ki | `forward`, lines 51-56; scale after each step | `_forward`, lines 348-359; log-sum-exp |
| Observation likelihood | log P(O) = sum_t log c_t with scaled recursion | `forward`, line 57 | `log_likelihood`, lines 176-178; log-sum-exp over final states |
| Viterbi recursion | delta_t(i) = B_i(o_t) max_k delta_(t-1)(k) A_ki | `viterbi`, lines 76-81 | `decode`, lines 265-276 |
| Backtracking | q*_t = phi_(t+1)(q*_(t+1)) | `viterbi`, lines 83-90 | `decode`, lines 278-289 |

## Differences that matter

Our Forward returns normalized filtered probabilities with shape `(T,N)`. Upstream `_forward` returns log joint probabilities with shape `(N,T)`. To compare, transpose the upstream table and normalize each time step in log space. Both compute the same observation log-likelihood, within floating-point tolerance.

Both Viterbi implementations use log-space, maximize over predecessor states, and backtrack. Upstream adds epsilon inside logarithms and replaces zero input probabilities with epsilon in its constructor; our implementation preserves exact zeros. Impossible sequences therefore have different semantics. The numerical parity checks deliberately use strictly positive probability tables, and pass copies to the reference constructor.

There are misleading words in the upstream `decode` docstring: line 214 describes a weighted sum, while the executable recurrence at line 275 takes a maximum. The returned score at line 289 is a log probability despite the return-description wording. Read the implementation together with the equations.

Our implementation supplies inference only. The HAR experiment estimates parameters using labeled training counts, not the upstream Baum-Welch `fit` routine. We do not claim to have implemented or numerically validated EM training.

## Numerical verification

Run `python -m experiments.part2_post_midterm.w09_hmm`. The harness verifies snapshot hashes, then extracts the unchanged `MultinomialHMM` class and the unchanged `logsumexp` helper with Python AST. This avoids importing unrelated numpy-ml package modules; it is not a full-package installation test.

Fifteen cases cover five seeds and sequence lengths 1, 6, and 1000. Maximum absolute differences were approximately 1.91e-13 for filtered state probabilities, 2.05e-11 for log-likelihood, and 3.87e-12 for Viterbi log-probability. Fourteen paths matched exactly. In the remaining case (seed 2, length 1000), different decoded paths had identical joint log-scores when recomputed under the same original parameters. Path equality is not required when multiple optimal paths tie.

The independent contract tests also compare small HMMs with exhaustive enumeration, so the reference implementation is not the only correctness check.

## HAR experiment design

The experiment uses only the original training partition: 5,270 fit windows from 15 subjects and 2,082 validation windows from six disjoint subjects. `GroupShuffleSplit(test_size=0.25, random_state=42)` preserves the Part I validation split. No official test file is loaded.

1. Preserve original row order. Join adjacent windows only when subject IDs agree and all nine inertial channels have matching overlapping halves (64 samples, absolute tolerance 1e-6, relative tolerance zero). Retain all windows, including singletons. Labels are not consulted for boundaries.
2. Fit StandardScaler and a 32-centroid MiniBatchKMeans on fit features only. Assign each window a categorical symbol by nearest centroid. The 32-symbol choice and add-one smoothing are fixed, without validation search.
3. Count initial states at segment starts, transitions inside verified fit segments, and state-symbol emissions on fit subjects. Normalize add-one-smoothed counts into pi, A, B.
4. Independent inference uses argmax_i pi_i B_i(o_t). Filtering and Viterbi use the same pi and B, with learned A. The control A with every row equal to pi must reproduce independent predictions; the script asserts this for every validation segment.
5. Repeat quantization with seeds 42, 0, 1, 2, 3 on the same subject split. Report all runs, accuracy, Macro-F1, confusion matrices, and saved per-window predictions. Seed variation is not subject-level uncertainty.

Source for the sampling scheme: [UCI HAR v1.0](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and the local dataset README: 128 samples/window, 50% overlap. The nine-channel overlap check is our conservative reconstruction, not a claim that the release supplies session timestamps.

## Findings and limitations

Mean validation Macro-F1 was 0.725746 for independent inference, 0.798413 for filtering, and 0.828615 for Viterbi. These are comparisons within this categorical-emission model, not a benchmark victory over other HAR classifiers. See [full results](../../results/w09_hmm/REPORT.md).

There are 424 fit segments and 176 validation segments. No verified link crosses an observed activity change. Consequently the experiment measures denoising inside stable activity segments; learned off-diagonal transitions are entirely supplied by smoothing. It does not establish transition-detection performance or justify stitching unrelated segments into one subject-wide sequence.

For seed 42, Viterbi fixes 283 independent errors but introduces 103 errors. Standing-to-sitting confusion accounts for 90 windows in the Viterbi confusion matrix; the first five saved error examples belong to this pattern. A persistent wrong interpretation can dominate an entire segment. Viterbi also uses future observations within the segment, whereas filtering is causal. Overlapping windows make emissions dependent even conditional on the activity, limiting likelihood calibration.
