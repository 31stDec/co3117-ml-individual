# W13 - LDA Review and Corrections

Reviewed on 2026-11-24 against the two-page [handwritten scan](w13-first-attempt.pdf), archived at `4c5fa6a`. The original scan is unchanged. Its commit identifies the reviewed artifact, not pre-AI provenance.

## Q1 - PCA vs LDA

The comparison and answers are completely correct.

- **Class labels:** LDA is a supervised dimensionality reduction method requiring class labels $y$, whereas PCA is unsupervised and uses only the input features $X$.
- **Preservation objective:** PCA seeks orthogonal projection directions that maximize the total variance of the projected data. LDA seeks projection directions that maximize class separability by maximizing the ratio of between-class variance to within-class variance (the Fisher criterion).
- **Why high variance can be poor for class separation:** Variance measures the global dispersion of data points regardless of class identity. If classes overlap significantly along the direction of maximum total variance (e.g., elongated parallel clusters), projecting onto that axis preserves large variance but collapses the classes together, resulting in severe class overlap. Conversely, a direction with smaller variance may cleanly separate the classes.

## Q2 - Fisher Criterion

The criterion definition and explanation are completely correct.

- **Fisher criterion:**
  $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
- **Numerator ($w^T S_B w$):** Represents the projected between-class scatter (the separation between the projected class centroids).
- **Denominator ($w^T S_W w$):** Represents the projected within-class scatter (the sum of variances/spread within each individual class along $w$).
- **Optimization:** We **maximize** $J(w)$ to ensure the projected class centers are pushed as far apart as possible while keeping each class cluster as compact as possible.

## Q3 - LDA Calculation

All calculation steps and numerical values are 100% correct.

Given:
$$m_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad m_2 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}, \quad S_W = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$$

1. **Inverse of within-class scatter matrix:**
   $$S_W^{-1} = \begin{pmatrix} \frac{1}{2} & 0 \\ 0 & 1 \end{pmatrix}$$
2. **Difference of class means:**
   $$m_2 - m_1 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$
3. **LDA direction:**
   $$w = S_W^{-1}(m_2 - m_1) = \begin{pmatrix} \frac{1}{2} & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 2 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$
4. **Unit normalization:**
   $$\|w\| = \sqrt{1^2 + 2^2} = \sqrt{5} \implies \hat{w} = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$
5. **Projected class means:**
   $$\hat{w}^T m_1 = \frac{1}{\sqrt{5}}(0) + \frac{2}{\sqrt{5}}(0) = 0$$
   $$\hat{w}^T m_2 = \frac{1}{\sqrt{5}}(2) + \frac{2}{\sqrt{5}}(2) = \frac{6}{\sqrt{5}}$$

**Geometric note:** The difference vector $m_2 - m_1 = (2, 2)^T$ lies along the $45^\circ$ line. Because feature 1 has higher within-class variance ($S_W[0,0]=2$) than feature 2 ($S_W[1,1]=1$), LDA down-weights feature 1 ($w_1 = 1$) and up-weights the more compact feature 2 ($w_2 = 2$). The resulting optimal direction $\hat{w} \propto (1, 2)^T$ tilts toward the feature with smaller spread, illustrating how LDA accounts for class covariance.

## Q4 - Number of Dimensions & Data Leakage

Both explanations are precise and mathematically sound.

- **Maximum LDA dimensions:**
  For $D = 561$ features and $C = 6$ classes, the maximum number of LDA dimensions is:
  $$\min(D, C - 1) = \min(561, 6 - 1) = 5$$
  **Explanation using $\text{rank}(S_B)$:** The between-class scatter matrix is defined as $S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$. Since the $C$ mean-deviation vectors $(\mu_c - \mu)$ satisfy $\sum_{c=1}^C N_c (\mu_c - \mu) = 0$, at most $C - 1$ of these vectors are linearly independent. Therefore, $\text{rank}(S_B) \le C - 1 = 5$, producing at most 5 non-zero eigenvalues in the generalized eigenvalue problem $S_B w = \lambda S_W w$.
- **Data leakage:** LDA is a supervised algorithm that uses both features $X$ and ground-truth labels $y$. If fitted on validation or test sets, the projection matrix encodes information about the test distribution and test labels. This causes target leakage, yielding an overfitted projection and artificially optimistic validation/test metrics. To maintain strict evaluation integrity, LDA must be fitted on training data only and applied as a fixed linear transform to evaluation data.

## Review Outcome

- **Calculations:** 100% correct across all questions (Q1–Q4). No arithmetic or conceptual mistakes found.
- **Presentation:** Explanations are clear, concise, and adhere strictly to course definitions and linear algebra principles.
