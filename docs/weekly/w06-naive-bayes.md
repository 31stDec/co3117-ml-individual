# W06 — Bayesian Learning / Naive Bayes

> **Course Week:** 6 | **Date:** 2026-10-06
> **Syllabus:** Chapter 6 (Tom Mitchell 1997) — Bayesian Learning / Naive Bayes
> **Depth:** A (BUILD)
> **Links:** [naive_bayes.py](../../src/from_scratch/naive_bayes.py)

---

## A. Concept Capsule

**Bayes' theorem** provides the foundation: $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$. The **Naive Bayes** classifier assumes conditional independence among features given the class label, decomposing the joint likelihood as $P(X|Y=c) = \prod_{j=1}^{d} P(x_j|Y=c)$.

1. **Gaussian Naive Bayes** models each $P(x_j | Y=c)$ as a univariate Gaussian $\mathcal{N}(\mu_{cj}, \sigma^2_{cj})$, estimated from training data with `ddof=0`. A smoothing term $\epsilon = \texttt{var\_smoothing} \times \max_j \operatorname{Var}(X_{\cdot j})$ is added to all variances to prevent division by zero and improve numerical stability.
2. **Classification** selects $\hat{y} = \arg\max_c \left[\log P(c) + \sum_j \log P(x_j | c)\right]$, avoiding floating-point underflow by working in log-space throughout.

---

## B. One Derivation or Worked Example

For a two-class problem with one feature $x$, class priors $P(c_0) = P(c_1) = 0.5$, $\mu_0 = 0, \sigma^2_0 = 1$ and $\mu_1 = 2, \sigma^2_1 = 1$:

Given $x = 1.5$:
$$\log P(x|c_0) = -\frac{1}{2}\left[\log(2\pi) + \frac{(1.5-0)^2}{1}\right] = -0.5[\log(2\pi) + 2.25] \approx -2.044$$
$$\log P(x|c_1) = -\frac{1}{2}\left[\log(2\pi) + \frac{(1.5-2)^2}{1}\right] = -0.5[\log(2\pi) + 0.25] \approx -1.044$$

Since $\log P(c_0) = \log P(c_1)$, the posterior favours $c_1$ by a log-odds margin of $1.0$. After normalisation: $P(c_1|x) \approx 0.731$.

---

## C. Code-to-Theory Trace

In [`src/from_scratch/naive_bayes.py`](../../src/from_scratch/naive_bayes.py):
* `fit()`: Computes per-class mean (`theta_`), variance (`var_`, ddof=0), and prior (`class_prior_`). Adds `epsilon_ = var_smoothing * X.var(axis=0).max()` to all variances.
* `predict_log_proba()`: Computes joint log-likelihood $\log P(c) + \sum_j \log P(x_j|c)$, then normalises via log-sum-exp to yield normalised log-posteriors (not necessarily calibrated).
* `predict()`: Returns `classes_[argmax(log_proba)]`.

---

## D. One Controlled Experiment

Contract comparisons below pass. A 14 Oct HAR benchmark fits 5,270 windows from 15 subjects and validates on 2,082 windows from six other train subjects: Macro-F1 0.7224, accuracy 0.7363, and 100% sklearn prediction agreement. Final test data is not used. Details: `results/naive_bayes_validation_audit.json`.

| Check | Result |
|---|---|
| `classes_` match | ✓ identical |
| `predict_log_proba` — `np.allclose(atol=1e-4)` | ✓ passes |
| Posteriors sum to 1.0 | ✓ `np.allclose(exp(L).sum(1), 1.0)` |
| `predict` labels match | ✓ 100% agreement |

---

## E. Failure / Misconception

* **14 Oct audit correction:** The earlier code scaled smoothing by the largest within-class variance. The required formula uses the largest feature variance over all training rows, including between-class variation. A new test with widely separated class means catches this difference. On HAR validation, 264 STANDING windows become SITTING. Mean maximum probability is 0.9971 despite accuracy 0.7363 (log loss 7.1509): normalisation does not ensure calibration.

---

## F. Written-Exam Capsule

> Naive Bayes classifies by computing the MAP estimate under a conditional-independence assumption: $\hat{y} = \arg\max_c P(c) \prod_j P(x_j|c)$. Gaussian NB uses per-class normal distributions $\mathcal{N}(\mu_{cj}, \sigma^2_{cj})$ with population variance (ddof=0). Variance smoothing adds $\epsilon = \texttt{var\_smoothing} \times \max_j \operatorname{Var}(X_{\cdot j})$ to prevent singularity. Despite the "naive" independence assumption, GNB is competitive on many real datasets because the decision boundary only requires correct posterior ranking, not calibrated probabilities.

---

## G. Reflection

* **Gained Insight:** Variance smoothing is critical for stability and must scale with the data's variance magnitude, not just be a fixed constant.
* **Next Target:** W07 Genetic Algorithms and Part I closeout, including hw4 (Bayesian networks) and hw5 (GA).

---

## H. Inquiry Trail

1. **Question:** How to replicate sklearn's GaussianNB log-posterior normalization exactly.
2. **Pre-AI Evidence:** Implemented initial version with raw NumPy before consulting references.
3. **Verification Source:** sklearn source code (`sklearn.naive_bayes.GaussianNB`) and Tom Mitchell (1997) Chapter 6.
4. **Reproduced:** Built numerically stable logsumexp normalization and variance smoothing from scratch, passing all contract tests with atol=1e-4 against sklearn.
