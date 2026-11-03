# W10 - Linear SVM reference study and HAR experiment

Date: 2026-11-03. This is a reference study and sklearn experiment, not a claim of implementing a new SVM solver. The handwritten drill and its corrections are separate work.

## Sources inspected

- HCMUT SVM lecture slides, 11 August 2026, slide numbers 5-13 (geometry and hard margin) and 34-35 (soft margin), local `C:/Users/USER/Downloads/ML/SVM.pdf`. Slide numbers differ from PDF page numbers because some slides have incremental overlays.
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch/tree/a2806c6732eee8d27762edd6d864e0c179d8e9e8), commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`, `mlfromscratch/supervised_learning/support_vector_machine.py` and `mlfromscratch/utils/kernels.py`. Exact source, MIT license, and SHA-256 hashes are saved in [svm/](svm/provenance.json).
- [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and installed sklearn 1.9.1 source, `svm/_classes.py::SVC`, `svm/_base.py::BaseSVC._get_coef` and `predict`. Source hashes are recorded in the experiment audit. The estimator delegates optimization to libsvm; the native solver was not reimplemented or audited.

## Mathematical formulation

For binary labels t in {-1,+1}, the decision score is f(x)=w dot x+b. Canonically scaled hard-margin SVM minimizes half the squared norm subject to t_i f(x_i)>=1. The distance from the boundary to either supporting plane is 1/||w||; the full width between those planes is 2/||w||.

Soft margin allows slack:

$$\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_i\xi_i,
\quad t_i(w^Tx_i+b)\geq1-\xi_i,\quad\xi_i\geq0.$$

At the optimum for fixed w,b, slack is hinge loss, max(0,1-t_i f(x_i)). A correctly classified point can still violate the margin. Increasing C makes violations more expensive relative to the norm penalty; it does not guarantee better generalization.

The dual maximizes:

$$\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jt_it_jK(x_i,x_j),
\quad0\leq\alpha_i\leq C,\quad\sum_i\alpha_i t_i=0.$$

Equivalently a quadratic-program solver minimizes the negative of this expression. For a linear kernel, w=sum_i alpha_i t_i x_i; only nonzero multipliers contribute to prediction.

## Equation-to-code map

Line numbers refer to the pinned local `svm/support_vector_machine.py` snapshot.

| Formula or operation | Reference location | Meaning |
|---|---|---|
| K_ij = x_i dot x_j | `fit`, lines 56-59; `kernels.py` lines 4-7 | Build the Gram matrix; the linear kernel returns an inner product. |
| Minimize 0.5 alpha^T Q alpha - sum alpha | `fit`, lines 62-63 and 79 | Q_ij=t_i t_j K_ij, q=-1; CVXOPT solves the quadratic program. |
| sum alpha_i t_i = 0 | `fit`, lines 64-65 | Equality constraint imposed by the solver. |
| 0 <= alpha_i <= C | `fit`, lines 71-76 | Stack -I and I for lower and upper bounds. |
| Nonzero multipliers define support vectors | `fit`, lines 82-92 | Keep multipliers above 1e-7. |
| f(x)=sum alpha_i t_i K(x_i,x)+b | `predict`, lines 104-110 | Compute signed score, then take its sign. |

In [our experiment](../../experiments/part2_post_midterm/w10_svm.py), lines 67-72 perform the subject split and fit-only scaling; lines 94-97 call linear SVC. Lines 106-109 reconstruct pairwise Xw+b scores and compute pairwise widths. The explicit objective and dual consistency checks are in `binary_checks`, lines 41-60.

## Important qualifications in the reference

The upstream class is binary and expects labels -1/+1. It cannot directly handle the six HAR labels. Its constructor defaults to an RBF kernel, so it should not be described as a linear model without changing that argument.

The intercept calculation at lines 94-98 uses the first support vector. For soft margin, the equality t_i f(x_i)=1 is guaranteed for a free support vector with 0<alpha_i<C, not every support vector. A point at alpha_i=C can violate the margin. A robust implementation uses free support vectors when available, or appropriate KKT bounds when none exist. This is a source-level observation, not an empirical claim of failure on a particular dataset.

Also, `if not self.C` treats zero like a hard-margin option. That is an implementation convention, not the standard limiting interpretation of the soft-margin objective as C approaches zero. Our experiment uses strictly positive C values.

The reference's CVXOPT solver is not executed here. HAR metrics come from sklearn SVC, which trains 15 one-versus-one classifiers for six classes. `decision_function_shape='ovo'` exposes their scores; taking argmax over those 15 scores would not produce a six-class prediction. The script uses `predict` for multiclass voting.

## Controlled experiment and checks

Grid fixed before execution: C={0.001,0.01,0.1,1,10}. All other settings, 561 features, and the previous 15/6 subject split remain fixed. StandardScaler is fitted on 5,270 fit windows, then applied to 2,082 validation windows. Only original training files are loaded; official test files are untouched. The grid is selected on validation Macro-F1, breaking exact ties toward smaller C. No RBF comparison is included before W11.

On a separate small binary example, box constraints and sum alpha_i t_i=0 pass; reconstructed decisions match sklearn, and the primal-dual gap is approximately 9.75e-10. Across all five HAR models, explicit linear scores match the 15 pairwise decision outputs with maximum absolute difference below 5.88e-11. Every solver reports convergence. These checks validate formula/output interpretation, not numerical parity with the unexecuted upstream solver.

## Results and limitations

See [report](../../results/w10_svm/REPORT.md), [audit](../../results/w10_svm/audit.json), and [metrics](../../results/w10_svm/metrics.csv). C=0.01 gives the best validation Macro-F1, 0.947470, accuracy 0.951969, with 1,036 support vectors. C=10 reaches training Macro-F1 1.0 but validation Macro-F1 drops to 0.936840. Stronger fitting is not automatically better validation performance.

The largest directed error at selected C is WALKING_DOWNSTAIRS to WALKING_UPSTAIRS, 42 windows; STANDING to SITTING contributes 23. Five individual errors and every prediction are saved. Similar activity features are a plausible explanation, but the confusion counts alone do not establish a causal explanation.

This is one tuning holdout, not an unbiased final estimate. Overlapping windows are correlated. The 15 widths are measured in standardized feature coordinates and are not one global multiclass margin. Timings are single-run measurements. No repeated seed sweep is claimed because this fixed, uncalibrated linear SVC fit is deterministic.
