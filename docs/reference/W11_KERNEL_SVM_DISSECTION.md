# W11 - Kernel SVM reference study

Prepared on 2026-11-09. This study extends W10 using its existing, pinned MIT reference snapshot. The handwritten drill and personal reflection remain separate work.

## Sources and scope

- HCMUT `SVM.pdf`, 11 August 2026: slide numbers 23-27 (dual representation), 28-33 (feature mapping and kernel trick), and 34-35 (soft margin). PDF page numbers differ from slide labels because of overlays. The lecture uses a polynomial example; the RBF sensitivity study is an assignment extension.
- ML-From-Scratch commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`: [kernel helpers](svm/kernels.py), [SVM solver source](svm/support_vector_machine.py), and [hashes/license provenance](svm/provenance.json).
- Installed sklearn SVC and `sklearn.metrics.pairwise.rbf_kernel`; the run records its package version and SVC source hash. Documentation: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [RBF kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html).

## Equations and source mapping

The binary soft-margin dual maximizes

$$\sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jt_it_jK(x_i,x_j),
\quad 0\leq\alpha_i\leq C,\quad\sum_i\alpha_it_i=0.$$

Prediction uses $f(x)=\sum_i\alpha_it_iK(x_i,x)+b$. A valid kernel computes an inner product in a feature space without explicitly constructing the feature vectors. The resulting boundary is linear in feature space and can be nonlinear in input space. Mapping alone does not guarantee useful separation or generalization.

The lecture's displayed dual on slides 24-25 has the maximization form despite the caption saying "to be minimized". A minimization QP uses its negative, as the reference code does.

| Operation | Pinned source | Interpretation |
|---|---|---|
| Linear kernel | `kernels.py`, `linear_kernel` | Inner product in the original input space. |
| Polynomial kernel | `kernels.py`, `polynomial_kernel` | `(inner(x1,x2)+coef)**power`; the lecture example uses power=2, coef=1. |
| RBF kernel | `kernels.py`, `rbf_kernel` | `exp(-gamma * ||x1-x2||^2)`. For distinct points, larger gamma reduces similarity. |
| Gram matrix | `support_vector_machine.py`, `fit` | Evaluate the chosen kernel for every training pair. |
| QP | `fit`, P/q/A/G construction | Multiply the Gram matrix by the label outer product; impose equality and box constraints. |
| Prediction | `predict` | Sum signed support-vector kernel contributions and add the intercept. |

C controls the penalty on margin violations; gamma controls how local RBF similarity is. They interact and should be tuned together on validation data. Distances depend on feature scaling, so the scaler must be fitted only on fit rows. Explicit numeric gamma values avoid conflating the experiment with automatic gamma conventions.

## Reference limitations

The upstream implementation is binary and expects labels -1/+1. HAR needs six classes; sklearn SVC handles this with 15 pairwise classifiers and voting. The experiment uses `predict`, not argmax over 15 binary scores.

The upstream default gamma is 1/n_features, and `if not self.gamma` treats zero as a request for that default. Its first-support-vector intercept calculation has the soft-margin caveat documented in W10: a free vector with 0<alpha<C satisfies the margin equality, whereas a bound vector need not. The experiment executes the small kernel helper module only; the CVXOPT solver is studied, not executed or claimed as our implementation.

## Experiment and numerical checks

[The script](../../experiments/part2_post_midterm/w11_kernel_svm.py) fixes C={0.1,1,10} and gamma={0.0001,0.001,0.01,0.1} before its first run. It retains W10's 15 fit subjects, six validation subjects, 561 features, and fit-only StandardScaler. Only the original training partition is loaded.

`kernel_checks` verifies saved source hashes, compares executed upstream RBF helpers to sklearn, checks small Gram matrices, reconstructs a binary SVC score from its dual coefficients, checks alpha constraints, and compares ordinary and precomputed-kernel SVC predictions. A separate explicit polynomial feature map checks the lecture identity. This verifies mathematical interpretation and helper parity, not full parity with the upstream optimizer.

The selected W10 linear model is fitted again, and every validation prediction must match the archived W10 predictions. GaussianNB with fixed smoothing and a majority classifier are rerun on the same rows as additional baselines. They are explicitly identified as new comparison runs. HMM sequence results are not mixed into an independent-window ranking.

See the generated [report](../../results/w11_kernel_svm/REPORT.md), [audit](../../results/w11_kernel_svm/audit.json), and [metrics](../../results/w11_kernel_svm/metrics.csv). Selection uses validation Macro-F1, then smaller C and gamma on exact ties. This reuses a tuning holdout; it is not an unbiased test evaluation. All timings are single-run observations, and the unequal linear/RBF search budgets limit kernel-ranking claims.
