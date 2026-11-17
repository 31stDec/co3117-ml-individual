# W12 - PCA review and clarifications

Reviewed on 2026-11-17 against the two-page [handwritten scan](w12-first-attempt.pdf), archived at `94f024a`. The original scan is unchanged. Its commit identifies the reviewed artifact, not pre-AI provenance.

## Q1 - Curse of dimensionality

The two main points, sparsity and overfitting with limited data, are appropriate. Dimensionality reduction can reduce computational cost and overfitting, but the wording about retaining the "most informative variance" needs qualification.

PCA retains directions with the greatest variance, not necessarily the most useful directions for predicting labels. Low-variance directions can contain class information; high-variance directions can contain noise. PCA also creates combinations of features rather than simply removing individual redundant features. Its benefit depends on the dataset and retained dimension.

Distances becoming less informative in high dimensions is a common issue, not a claim that every distance measure fails on every high-dimensional dataset.

## Q2 - Variance maximization and decorrelation

The objective and constraint are correct:

$$\max_u u^T S u\quad\text{subject to}\quad u^T u=1.$$

The covariance argument is also correct. For orthonormal eigenvectors,

$$\operatorname{Cov}(z_i,z_j)=u_i^T S u_j=\lambda_j u_i^T u_j=0\quad(i\ne j).$$

If eigenvalues repeat, choose an orthonormal basis within the repeated eigenspace. Uncorrelated does not generally mean statistically independent. Decorrelation refers to the covariance used to fit PCA; new validation data need not have exactly diagonal score covariance.

## Q3 - Small PCA calculation

All calculations are correct. The characteristic equation is (3-lambda)^2-1=0, so the eigenvalues are 4 and 2. A unit first direction is (1,1)/sqrt(2), and retained variance is 4/(4+2)=2/3, approximately 66.7%. The negative of this direction is equally valid and gives the same projection subspace.

## Q4 - Projection and reconstruction

The answers are correct: x-m=(2,0), z=sqrt(2), reconstructed x=(2,2), and squared reconstruction error is 1^2+(-1)^2=2. This is squared Euclidean error for one point, not mean squared error per feature.

The leakage explanation is correct. Fit the mean, any scaler, and PCA directions on training rows, then reuse them for validation/test. PCA is unsupervised, but fitting it on validation/test features still exposes evaluation-distribution information under this assignment's protocol.

## Implementation convention

The course code interface requires sample explained variances with denominator n-1. With centered data Xc=U diag(s) V^T, the component rows are V^T and explained_variance_i=s_i^2/(n-1). A 1/n convention changes the variance scale, but not the directions or retained-variance ratios. The code uses n-1 consistently.

Verification: direct recalculation, the HCMUT PCA lecture, course PCA contract, and NumPy/sklearn checks recorded in `results/w12_pca/audit.json`. No new closed-book reproduction is claimed.
