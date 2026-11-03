# W10 - SVM: Maximum and Soft Margin

Date: 2026-11-03. Depth: B reference study and C experimental comparison.

**Question I am trying to answer:** How does C affect fitting and validation performance for a linear SVM on HAR?

Working hypothesis, recorded retrospectively: larger C should favor fitting the training data more closely. Validation performance may peak at an intermediate value. This is not a timestamped pre-experiment hypothesis.

## A. Concept capsule

A binary SVM predicts from the sign of f(x)=w^T x+b. Hard margin assumes linear separability and minimizes half the squared norm under t_i f(x_i)>=1. Soft margin introduces nonnegative slack:

$$\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_i\xi_i,
\qquad t_i(w^Tx_i+b)\geq1-\xi_i.$$

The minimum slack is hinge loss, max(0,1-t_i f(x_i)). Larger C makes violations more expensive relative to the norm penalty.

## B. One derivation or worked example

For w=(1,0) and b=-2, the decision boundary is x_1=2. The points (1,0) and (3,0) lie on the supporting planes. Since ||w||=1, the one-sided margin is 1 and the full width is 2. With positive labels and scores 0.4 and -0.5, the minimum slacks are 0.6 and 1.5. Only the second sample is misclassified.

The [scan](../../exercises/w10-first-attempt.pdf) contains correct arithmetic. The [corrections](../../exercises/w10-corrections.md) clarify margin terminology and statements about C.

## C. Code-to-theory trace

The pinned [reference study](../reference/W10_SVM_DISSECTION.md) maps the dual quadratic term and linear term to upstream `fit` lines 62-63, the equality constraint to lines 64-65, and the multiplier bounds to lines 71-76. Prediction sums support-vector contributions in lines 104-110.

The [HAR script](../../experiments/part2_post_midterm/w10_svm.py) uses sklearn SVC rather than the upstream CVXOPT solver. Lines 106-109 verify pairwise linear scores and compute widths. A separate binary check verifies the dual constraints and obtains a primal-dual gap below 1e-9. There is no claim of implementing a new SVM solver.

## D. One controlled experiment

The split preserves 5,270 fit windows from 15 subjects and 2,082 validation windows from six other subjects. StandardScaler sees only fit data. All 561 features are retained. The official test set is not read.

Only C varies; the linear kernel and other settings remain fixed. Six classes use fifteen one-versus-one classifiers. Probability calibration is disabled, so repeated seed runs on the same split are omitted.

| C | Fit Macro-F1 | Validation Macro-F1 |
|---|---:|---:|
| 0.001 | 0.973240 | 0.938075 |
| 0.01 | 0.985773 | 0.947470 |
| 0.1 | 0.993487 | 0.941960 |
| 1 | 0.997600 | 0.935936 |
| 10 | 1.000000 | 0.936840 |

C=0.01 is selected by validation Macro-F1, with accuracy 0.951969 and 1,036 support vectors. [Full results](../../results/w10_svm/REPORT.md) contain timings, confusion matrices, and predictions. Selection on this holdout is not an unbiased final evaluation.

## E. Failure / misconception

At selected C, 42 WALKING_DOWNSTAIRS windows are predicted as WALKING_UPSTAIRS, and 23 STANDING windows as SITTING. Increasing C improves training fit but not validation monotonically. The upstream bias calculation also needs care: a soft-margin support vector at its upper multiplier bound need not satisfy the margin equality.

## F. Written-exam capsule

Support vectors have nonzero dual multipliers. The one-sided canonical margin is 1/||w||; the full width is twice that. Slack permits margin violations. Hinge loss can be positive even for correctly classified points. C controls the trade-off between violations and the norm penalty. Validation is needed because stronger fitting does not guarantee better generalization.

## G. Reflection

The numerical example distinguishes correct classification from satisfying the margin. The experiment distinguishes training fit from held-out performance. Remaining work is to study kernel SVMs in W11; this linear study does not establish whether RBF will improve results.

## H. Inquiry trail

Codex assisted question preparation, handwritten review, reference reading, experiment scripting, and documentation. The scan is archived at `358a662`; it identifies the reviewed artifact, not pre-AI provenance. Verification used the lecture slides, pinned reference source, installed sklearn, and numerical checks. Closed-book reproduction has not been recorded for this session. Details are in [AI_USE.md](../../AI_USE.md).
