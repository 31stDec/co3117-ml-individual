# W06 Written Drill — Review and Corrections

> **Reference Artifact:** [`exercises/w06-first-attempt.pdf`](w06-first-attempt.pdf)
> **First-Attempt Commit:** `daa90c0`
> **Textbook Reference:** Tom Mitchell (1997), *Machine Learning*, Chapter 6 (Bayesian Learning).

---

## 1. Item-by-Item Verification

### Q1: MAP vs ML Estimation
- **Evaluation:** Likelihoods computed correctly: $P(D|\theta=0.6) = 0.6^2 \times 0.4 = 0.144$, $P(D|\theta=0.5) = 0.5^3 = 0.125$. Unnormalised posteriors: $0.1008$ vs $0.0375$. MAP = $\theta=0.6$.
- **Citation:** Mitchell (1997), Section 6.2 (Bayes Theorem and Concept Learning).

### Q2: Naive Bayes Discrete Classification
- **Evaluation:** Conditional independence correctly applied: $P(c_1|x) \propto 0.5 \times 0.7 \times 0.2 = 0.07$, $P(c_2|x) \propto 0.5 \times 0.4 \times 0.6 = 0.12$. Normalised posteriors $0.368 / 0.632$. Prediction $c_2$ correct.
- **Citation:** Mitchell (1997), Section 6.9 (Naive Bayes Classifier).

### Q3: Variance Smoothing
- **Evaluation:** Correctly identifies division-by-zero with $\sigma^2 = 0$. Smoothing formula $\epsilon = \texttt{var\_smoothing} \times \max(\sigma^2)$ correctly stated with numerical example ($4 \times 10^{-9}$).
- **Citation:** sklearn documentation (`sklearn.naive_bayes.GaussianNB`); Mitchell (1997) Chapter 6.

### Q4: Conditional Independence Assumption
- **Evaluation:** Independence assumption correctly stated. Text classification example (word co-occurrence) is a valid violation. Explanation citing Domingos & Pazzani (1997) is accurate — ranking robustness explains NB's empirical success.
- **Citation:** Mitchell (1997), Section 6.9; Domingos & Pazzani (1997), "On the Optimality of the Simple Bayesian Classifier under Zero-One Loss."

---

## 2. Summary & Score
- **Total Score:** 4 / 4 subquestions verified and accurate.
- **No fundamental errors detected.**
