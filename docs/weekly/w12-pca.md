# W12 - PCA and the curse of dimensionality

Date: 2026-11-17. Depth A implementation and reference comparison. Prepared for review before commit.

**Question I am trying to answer:** How many HAR dimensions can PCA remove while preserving reconstruction quality and classification performance?

Study expectation, recorded retrospectively: keeping more components should reduce reconstruction error, but preserving variance need not preserve class information. This is not a recorded pre-AI hypothesis.

## A. Concept capsule

With limited samples, high-dimensional data can be sparse and models can overfit. PCA projects centered observations onto orthonormal directions of maximum variance. For covariance S, the first direction maximizes u^T S u subject to u^T u=1. Subsequent directions are orthogonal eigenvectors. Their scores are uncorrelated on the fitting data, but not necessarily independent. PCA does not use labels.

## B. One derivation or worked example

For S=[[3,1],[1,3]], the eigenvalues are 4 and 2. A unit first direction is (1,1)/sqrt(2), retaining 4/6 of the variance. With mean (1,1) and point (3,1), the score is sqrt(2), reconstruction is (2,2), and squared error is 2.

These calculations are correct in the [scan](../../exercises/w12-first-attempt.pdf), committed at `94f024a`. The [review](../../exercises/w12-corrections.md) clarifies that high variance does not guarantee predictive usefulness and that decorrelation differs from independence.

## C. Code-to-theory trace

In [NumPy PCA](../../src/from_scratch/pca.py), `fit_pca` centers X, computes thin SVD, and returns component rows from V^T. Explained variances are s_i^2/(n-1), following the course API. `transform` computes (X-mean)W^T; `inverse_transform` restores ZW+mean. No whitening is applied.

The pinned ML-From-Scratch reference centers its covariance calculation but projects uncentered X. It also refits on every `transform` call. The comparison therefore centers the example explicitly and never uses that API to refit validation data. Numerical checks compare projection subspaces, avoiding arbitrary eigenvector signs. sklearn full-SVD variance and reconstruction checks also pass.

## D. One controlled experiment

The [HAR experiment](../../experiments/part2_post_midterm/w12_pca.py) retains the previous 5,270 fit windows from 15 subjects and 2,082 validation windows from six subjects. StandardScaler and PCA see fit rows only. Official test files are not read. A linear SVC keeps C=0.01 from W10; only the retained dimension changes, without whitening or further scaling.

| Components | Retained fit variance | Validation MSE | Validation Macro-F1 |
|---|---:|---:|---:|
| 2 | 0.573421 | 0.427430 | 0.487450 |
| 20 | 0.778598 | 0.256718 | 0.881138 |
| 100 | 0.951285 | 0.063233 | 0.932203 |
| 200 | 0.994088 | 0.009074 | 0.943953 |
| 561 | 1.000000 | approximately 0 | 0.947470 |

[All eight settings and the baseline](../../results/w12_pca/metrics.csv) are saved. The no-PCA baseline reproduces W10 predictions. MSE averages over rows and all 561 standardized features. The fit spectrum reaches 95% variance at 99 components; 100 is the nearby tested grid point. Among tested reduced settings, 200 gives the best validation Macro-F1, but does not beat the baseline.

## E. Failure / misconception

Two components retain over half the variance yet classify poorly. PCA optimizes reconstruction, not label separation. Increasing k reduces reconstruction error, but a general monotonic improvement in classification is not guaranteed. This is one previously examined validation holdout with correlated windows, not an unbiased final-test estimate.

## F. Written-exam capsule

Center before projecting. Order eigenvalues from largest to smallest. Eigenvector signs are arbitrary. Uncorrelated scores need not be independent. Use training means and directions on new data. Distinguish squared reconstruction error from per-feature MSE, and n versus n-1 variance conventions.

## G. Reflection

The drill's arithmetic agrees with the implementation. The experiment shows why a variance threshold alone is insufficient for choosing a classifier representation. LDA's use of class information is the next comparison for W13.

## H. Inquiry trail

Codex assisted questions, review, PCA implementation, reference checks, experiments, and documentation. Verification used the lecture, pinned source, sklearn, and numerical tests. The scan hash identifies the reviewed artifact, not pre-AI provenance. Closed-book reproduction is not recorded; assistance is listed in [AI_USE](../../AI_USE.md).
