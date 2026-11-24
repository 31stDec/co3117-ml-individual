# W13 - Linear Discriminant Analysis and feature engineering

Date: 2026-11-24. Depth B/C study and controlled HAR evaluation. Prepared for review before commit.

**Question I am trying to answer:** Can supervised Linear Discriminant Analysis preserve HAR class separability better than unsupervised PCA when restricted to very low dimensions?

Study expectation, recorded retrospectively: LDA should outperform PCA at small dimensions because it optimizes between-class over within-class scatter, whereas PCA preserves total variance without label guidance.

## A. Concept capsule

PCA seeks orthogonal axes of maximal variance, ignoring labels. Linear Discriminant Analysis seeks projections that maximize class separation relative to within-class spread. It defines two scatter matrices: between-class $S_B$ and within-class $S_W$. Fisher's criterion maximizes the Rayleigh quotient:

$$J(w) = \frac{w^T S_B w}{w^T S_W w}.$$

This yields the generalized eigenvalue problem $S_B w = \lambda S_W w$. Because $S_B$ is built from $C$ mean differences that sum to zero, $\text{rank}(S_B) \le C - 1$. LDA extracts at most $C - 1$ components regardless of input dimension $D$.

## B. One derivation or worked example

For two classes with shared covariance, $w \propto S_W^{-1}(\mu_2 - \mu_1)$. With $m_1 = (0, 0)^T$, $m_2 = (2, 2)^T$, and $S_W = \operatorname{diag}(2, 1)$, the mean difference is $(2, 2)^T$. Inverting $S_W$ gives $\operatorname{diag}(1/2, 1)$.

Thus $w = (1, 2)^T$, with unit vector $\hat{w} = \frac{1}{\sqrt{5}}(1, 2)^T$. Projected means are $0$ and $6/\sqrt{5}$. Feature 1 has larger intra-class spread, so LDA down-weights it and tilts toward the tighter vertical axis.

These calculations match the [handwritten scan](../../exercises/w13-first-attempt.pdf) at `4c5fa6a` and [corrections](../../exercises/w13-corrections.md).

## C. Code-to-theory trace

In scikit-learn's `LinearDiscriminantAnalysis`, the SVD solver avoids computing or inverting $S_W$ directly. It spheres data using class covariances, then applies SVD to transformed centroids. When $n < D$, $S_W$ is singular; shrinkage regularizes $S_W$ toward an identity target.

## D. One controlled experiment

The [HAR experiment](../../experiments/part2_post_midterm/w13_lda.py) uses 5,270 fit and 2,082 validation windows from separate subjects. `StandardScaler` sees fit rows only. We compare LDA ($k \in \{1,\dots,5\}$) plus Linear SVC ($C=0.01$) against baseline SVC and PCA ($k=5$).

| Model | $k$ | Validation Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Baseline SVC | 561 | 0.947470 | 0.951969 |
| PCA + SVC | 5 | 0.815499 | 0.827089 |
| LDA + SVC | 1 | 0.530285 | 0.606628 |
| LDA + SVC | 3 | 0.776375 | 0.793468 |
| LDA + SVC | 5 | 0.966564 | 0.969260 |
| Direct LDA (SVD) | 5 | 0.964329 | 0.967819 |

[Metrics](../../results/w13_lda/metrics.csv) and [predictions](../../results/w13_lda/predictions.csv) are saved. At $k=5$, LDA + SVC reaches Macro-F1 0.966564, beating both PCA-5 (0.815499) and the full 561-feature baseline (0.947470).

## E. Failure / misconception

1. *High variance implies separability:* PCA preserves variance, but drops 13% Macro-F1 at $k=5$. Low-variance directions can carry key class boundaries.
2. *Arbitrary component count:* $\text{rank}(S_B) \le C - 1$. For 6 HAR classes, LDA can extract at most 5 components.
3. *Data leakage:* Fitting LDA on validation or test sets leaks label distributions into projection weights, inflating metrics.

## F. Written-exam capsule

- Fisher criterion: maximize $w^T S_B w / w^T S_W w$.
- Maximum components: $\min(D, C - 1)$.
- Two-class shortcut: $w^* = S_W^{-1}(\mu_2 - \mu_1)$.
- Singular $S_W$ ($N < D$): use shrinkage or PCA preprocessing.
- Fit projection strictly on training rows.

## G. Reflection

Supervised projection leverages class labels to discard hundreds of noisy dimensions. On HAR, 5 LDA dimensions beat all 561 raw features.

## H. Inquiry trail

Assisted with drill review (`4c5fa6a`), derivation checks, and HAR experiment scripting. Verified against HCMUT slides and Bishop PRML Section 4.1.4. Details recorded in [AI_USE](../../AI_USE.md).
