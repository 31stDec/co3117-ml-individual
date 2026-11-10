# W11 - Kernel SVM review and supplements

Reviewed on 2026-11-10 against the two-page [scan](w11-first-attempt.pdf), archived at `b01f02d`. The original scan is unchanged. This hash identifies the reviewed artifact, not pre-AI provenance.

## Q1 - Support vectors and margin

The calculation is correct. Dividing f(x)=2x-2 by two gives the canonical score g(x)=x-1, with w=1 and b=-1. The boundary is x=1. The supporting points x=0 and x=2 are each one unit from the boundary: one-sided margin 1, full width 2.

## Q2 - Dual prediction

The written score is correct:

$$f(x)=\sum_i\alpha_i y_i K(x_i,x)+b.$$

An alpha_i of zero means the point has no direct contribution to this fitted decision function. Saying it must lie **strictly** outside the margin is too strong. At an optimal standard soft-margin solution with C>0, alpha_i=0 implies y_i f(x_i)>=1; equality can occur in a degenerate case. Points strictly outside the margin have zero alpha, but the reverse need not be strict. This describes the fitted solution, not what will happen if the training data change and the model is refitted.

### Supplement - Why kernels avoid explicit features

This is an added study note, not counted here as an omitted answer.

A kernel computes K(x_i,x)=phi(x_i)^T phi(x) directly. The dual score needs only these inner products, so it can be evaluated without explicitly constructing phi(x), even in a very large feature space.

## Q3 - Two kernels

All calculations are correct: u dot v=0, the polynomial kernel equals 1, and ||u-v||^2=2. With gamma=0.5, RBF gives exp(-1), approximately 0.368. For these distinct points, increasing gamma decreases similarity. For identical points, RBF similarity remains 1.

## Q4 - SVM versus Logistic Regression

The loss comparison is correct. Standard soft-margin SVM uses max(0,1-yf(x)); binary Logistic Regression uses log(1+exp(-yf(x))). Logistic loss is positive at finite scores and approaches zero as the correct-class margin grows.

Two wording changes:

- Binary Logistic Regression outputs model probabilities through a sigmoid, but these are **not guaranteed to be calibrated** on new data.
- SVM's f(x) is a decision score. For a linear model, signed geometric distance is f(x)/||w||. The score itself should not be called geometric distance. Probability estimates require an additional estimation or calibration step.

The validation explanation is correct: using test labels to select C or gamma makes the final evaluation optimistic. The project already records historical test exposure; W11 reads only the original training partition and its subject-held-out validation split.

## Verification

HCMUT SVM slides cover canonical margins, the dual score, kernel trick, and soft-margin constraints. Direct recalculation confirms Q1/Q3. The [reference study](../docs/reference/W11_KERNEL_SVM_DISSECTION.md) and [audit](../results/w11_kernel_svm/audit.json) provide source mapping and numerical checks. No new closed-book reproduction is claimed.
