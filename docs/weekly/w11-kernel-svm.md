# W11 - Kernel SVM and model comparison

Prepared on 2026-11-09; reviewed on 2026-11-10. Depth B/C. Ready for student review before commit.

**Question I am trying to answer:** Does an RBF SVM improve on the W10 linear model under the same HAR validation protocol?

Study expectation: a nonlinear kernel can fit more complex boundaries, but its benefit depends on C and gamma. This is an AI-assisted study framing, not a recorded pre-AI hypothesis.

## A. Concept capsule

The kernel trick replaces an explicit feature-space inner product with K(x,z). For RBF, K(x,z)=exp(-gamma ||x-z||^2). Nearby standardized observations have higher similarity. Larger gamma makes similarity more local; C controls the penalty for margin violations relative to regularization. A more flexible boundary does not guarantee better performance on unseen subjects.

## B. One derivation or worked example

For a binary model, substitute w=sum_i alpha_i t_i phi(x_i) into the feature-space score. This gives f(x)=sum_i alpha_i t_i K(x_i,x)+b. Terms with alpha_i=0 vanish. Prediction needs kernel evaluations rather than explicit feature coordinates. The [scan](../../exercises/w11-first-attempt.pdf), archived at `b01f02d`, has correct margin and kernel arithmetic. The [review](../../exercises/w11-corrections.md) clarifies wording and adds the kernel-trick explanation as a supplement.

## C. Code-to-theory trace

The pinned reference's `rbf_kernel` computes squared Euclidean distance and applies the exponential. Its SVM `fit` builds the Gram matrix and the label-weighted quadratic term. The [reference study](../reference/W11_KERNEL_SVM_DISSECTION.md) explains the dual constraints and source limitations.

The [experiment](../../experiments/part2_post_midterm/w11_kernel_svm.py) executes the reference kernel helpers and uses sklearn SVC for HAR. Small numerical checks cover kernel parity, polynomial feature mapping, binary dual scores, and precomputed-kernel predictions. The upstream CVXOPT optimizer is not executed.

## D. One controlled experiment

The fixed grid contains C={0.1,1,10} and gamma={0.0001,0.001,0.01,0.1}. StandardScaler uses only 5,270 fit windows from 15 subjects; validation contains 2,082 windows from six other subjects. All 561 features remain, and official test files are not read.

The W10 linear model reproduces archived predictions exactly. GaussianNB and a majority classifier provide baselines. [Results](../../results/w11_kernel_svm/REPORT.md) include the grid, errors, and timings. Selection maximizes validation Macro-F1; smaller C, then gamma, break ties. Deterministic fits are not repeated across training seeds.

## E. Failure / misconception

The best tested RBF setting is C=10, gamma=0.0001: validation Macro-F1 0.944692 and accuracy 0.950528. Linear SVM scores 0.947470, GaussianNB 0.722378, and majority 0.052020. The RBF selection lies on the grid boundary, so it is only the best tested configuration, not an established optimum. At C=1, gamma=0.1, fit F1 reaches 1.0 while validation F1 falls to 0.052020.

A nonlinear kernel is not automatically superior. Local similarity can fit training observations but transfer poorly between subjects. Fit-only scaling is essential. Unequal tuning budgets and a reused holdout limit ranking claims. HMM scores are excluded because they use temporal context.

## F. Written-exam capsule

Distinguish input space from feature space and decision scores from probabilities. SVM uses hinge loss; Logistic Regression uses logistic loss and models class probabilities. Uncalibrated SVC scores are not probabilities. Neither a large C nor a large gamma guarantees improved validation accuracy. Choose hyperparameters without using the final test labels.

## G. Reflection

The review separates correct arithmetic from overly strong claims. Zero alpha does not require a point to be strictly outside the margin, and Logistic Regression probabilities are not automatically calibrated. The experiment likewise separates training fit from validation performance. These results support retaining the linear model for this comparison, without claiming it will always beat RBF.

## H. Inquiry trail

Codex assisted question preparation, handwritten review, reference study, experiment scripting, and documentation. The reviewed scan is archived at `b01f02d`; it is not evidence of pre-AI provenance. Verification uses the lecture, pinned code, installed sklearn, and numerical checks. Closed-book reproduction has not been recorded. The session is summarized in [AI_USE](../../AI_USE.md).
