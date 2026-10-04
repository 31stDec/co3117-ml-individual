# PRE-RELEASE CATCH-UP (W01–W04)

> **Author:** Pham Minh Tri (2353235)  
> **Written on:** 2026-09-24  
> **Covers:** Foundations & ML Workflow (Ch. 1), Decision Trees (Ch. 2)  
> **Reference Artifact:** [`exercises/release-baseline-w01-w04.pdf`](../../exercises/release-baseline-w01-w04.pdf)  

---

## A. Concept Capsule

### Foundations (Ch. 1): ML Workflow, Generalization, and Metrics
Supervised learning approximates a target function $f: \mathcal{X} \to \mathcal{Y}$ from labeled examples $(x_i, y_i)$.
* **Data Partitioning:** Data must be strictly divided into training, validation, and sealed test sets. Preprocessing parameters (e.g., mean and standard deviation) must be computed on training data only to prevent leakage.
* **Underfitting vs. Overfitting:** Underfitting occurs when model capacity is too restrictive (high bias, high training and generalization errors). Overfitting occurs when excessive capacity models training noise (low training error, high test error).
* **Bias-Variance Tradeoff:** Expected generalization error decomposes into:
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$
  Constrained models exhibit high bias; flexible models exhibit high variance.
* **Evaluation Metrics:** On multi-class imbalanced distributions, raw accuracy is uninformative. We prioritize Macro-averaged F1-score:
  $$\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^K F1_k$$

### Decision Trees (Ch. 2): Partitioning and Pruning
Decision trees recursively partition the feature space via orthogonal hyperplanes:
* **Impurity Measures:** Evaluated using Entropy $H(S) = -\sum p_k \log_2 p_k$ or Gini index $G(S) = 1 - \sum p_k^2$.
* **Information Gain:** The expected entropy reduction:
  $$IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$
* **Continuous Features:** Handled by sorting unique values and testing candidate midpoints.
* **Regularization:** Without early stopping (pre-pruning) or cost-complexity pruning (post-pruning), trees expand until leaves are pure, causing severe overfitting.

---

## B. One Derivation or Worked Example

We calculate the root split on *Play Tennis* ($|S|=14$, 9 positive, 5 negative):
1. **Root Entropy:**
   $$H(S) = -(9/14 \log_2 9/14 + 5/14 \log_2 5/14) \approx 0.9402 \text{ bits}$$
2. **Conditional Entropy for Wind $\in \{\text{Weak}, \text{Strong}\}$:**
   * $S_{\text{Weak}}$ (8 instances: 6+, 2-): $H(S_{\text{Weak}}) \approx 0.8113 \text{ bits}$
   * $S_{\text{Strong}}$ (6 instances: 3+, 3-): $H(S_{\text{Strong}}) = 1.0000 \text{ bits}$
3. **Information Gain:**
   $$IG(S, \text{Wind}) = 0.9402 - [8/14(0.8113) + 6/14(1.0)] = 0.0480 \text{ bits}$$
Positive gain confirms uncertainty reduction.

---

## C. Code-to-Theory Trace

In [`src/metrics.py`](../../src/metrics.py) and [`src/from_scratch/tree_split.py`](../../src/from_scratch/tree_split.py):
* `entropy()` and `gini()` compute node impurity directly from label distributions.
* `information_gain()` evaluates candidate boolean masks $x \le t$.
* `best_threshold()` searches midpoints between adjacent sorted features.
* `macro_f1()` computes unweighted harmonic means across classes.

---

## D. Controlled Experiment: Release Baseline

On UCI HAR (561 features, 6 classes, subject-aware split), we evaluated naive baselines:

| Baseline Strategy | Test Accuracy | Test Macro-F1 | Observation |
|---|---|---|---|
| **Majority-Class** (Always predict `LAYING`) | 18.22% | **0.0440** | Zero recall on all minority classes |
| **Stratified Random** (Prior distribution) | 16.39% | **0.1387** | Chance baseline across all classes |

Logged in `results/metrics.csv`. Any valid model must surpass Macro-F1 $= 0.14$.

---

## E. Failure / Misconception

* **Misconception:** High training accuracy indicates a healthy decision tree.
* **Reality:** Unpruned trees easily memorize noise, achieving 100% training accuracy while collapsing on test data due to high variance.

---

## F. Written-Exam Capsule

> Increasing model complexity decreases bias but increases variance. Optimal generalization occurs where marginal bias reduction equals marginal variance increase. Decision trees construct axis-aligned splits by maximizing information gain. Without pre-pruning or post-pruning, trees overfit by partitioning until leaf purity. On imbalanced multi-class problems, Macro-F1 is essential because it weights classes equally, exposing trivial majority-class guessing.

---

## G. Reflection

* **Gained Competence:** Concrete understanding of impurity criteria, bias-variance tradeoffs, and the necessity of Macro-F1 for human activity recognition.
* **Next Target:** Implementing single-layer Perceptron and multi-layer Perceptron backpropagation in W05.

---

## H. Inquiry Trail

* **Question:** Minimum written drill requirements for pre-release baseline diagnostic.
* **Pre-AI Artifact:** Handwritten diagnostic [`exercises/release-baseline-w01-w04.pdf`](../../exercises/release-baseline-w01-w04.pdf) committed at `c521df4`.
* **Verification:** Mitchell (1997) Chapters 1–3.
