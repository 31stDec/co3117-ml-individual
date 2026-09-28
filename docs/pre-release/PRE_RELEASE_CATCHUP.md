# PRE-RELEASE CATCH-UP (W01–W04)

> **Author:** Pham Minh Tri (2353235)  
> **Created:** 2026-09-24 (Release week, Course Week 5)  
> **Covers:** Foundations & ML Workflow (Ch. 1), Decision Trees (Ch. 2)  
> **Reference Artifact:** [`exercises/release-baseline-w01-w04.pdf`](../../exercises/release-baseline-w01-w04.pdf)  
> **Word Count:** ~750 words  

---

## A. Concept Capsule

### Foundations (Ch. 1): ML Workflow, Generalization, and Metrics
Machine learning formulates task execution through data-driven parameter estimation rather than explicit rule programming. In **supervised learning**, we approximate a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from labeled tuples $(x_i, y_i)$. To guarantee empirical validity and prevent data leakage:
* **Dataset Partitioning:** Data must be partitioned into strict training, validation, and sealed test sets. Preprocessing statistics (e.g., mean, scaling factors) must be fit solely on the training partition and broadcast downstream.
* **Underfitting vs. Overfitting:** Underfitting occurs when hypothesis space capacity is insufficient to capture underlying data structure (high bias, high train error, high test error). Overfitting occurs when excessive capacity allows the model to interpolate noise and idiosyncrasies of the training distribution (low train error, high test error).
* **Bias-Variance Decomposition:** For squared loss, expected generalization error decomposes into:
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma_{\text{noise}}^2$$
  Increasing model complexity monotonically reduces bias while increasing variance. The optimal model balances this tradeoff at minimal total risk.
* **Evaluation Metrics:** For multi-class classification under class imbalance, raw accuracy is deceptive. We prioritize **Macro-averaged F1-score**, which calculates the unweighted mean of per-class harmonic means of precision and recall:
  $$\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

### Decision Trees (Ch. 2): Recursive Partitioning and Pruning
Decision Trees perform non-parametric recursive partitioning of the input space through axis-aligned orthogonal splits:
* **Impurity Criteria:** Node purity is evaluated using **Entropy** $H(S) = -\sum_{k=1}^K p_k \log_2 p_k$ or **Gini Impurity** $G(S) = 1 - \sum_{k=1}^K p_k^2$.
* **Information Gain:** The reduction in entropy achieved by partitioning sample set $S$ using candidate attribute $A$:
  $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
* **Continuous Features & Missing Values:** Continuous attributes are discretized dynamically by sorting values and evaluating midpoint thresholds maximizing impurity reduction. Missing attributes are addressed via mode/mean imputation or fractional instance weighting.
* **Pruning and Inductive Bias:** The unconstrained inductive bias of decision trees favors smaller, shallow trees over deeper ones (Occam's razor), yet unregularized trees expand until pure leaves memorizing all training noise. **Pre-pruning** halts expansion early via depth/split thresholds, whereas **post-pruning** (e.g., cost-complexity pruning) trims subtrees retroactively based on validation performance.

---

## B. One Derivation or Worked Example

We reproduce the manual derivation verified in handwritten artifact [`release-baseline-w01-w04.pdf`](../../exercises/release-baseline-w01-w04.pdf) for the canonical *Play Tennis* problem ($|S| = 14$ samples, 9 positive $Y$, 5 negative $N$):

1. **Initial Root Entropy:**
   $$H(S) = -\left(\frac{9}{14}\log_2 \frac{9}{14} + \frac{5}{14}\log_2 \frac{5}{14}\right) \approx 0.9402 \text{ bits}$$

2. **Conditional Entropy for Attribute `Wind` $\in \{\text{Weak}, \text{Strong}\}$:**
   * Subset $S_{\text{Weak}}$ (8 instances: 6 $Y$, 2 $N$):
     $$H(S_{\text{Weak}}) = -\left(\frac{6}{8}\log_2 \frac{6}{8} + \frac{2}{8}\log_2 \frac{2}{8}\right) \approx 0.8113 \text{ bits}$$
   * Subset $S_{\text{Strong}}$ (6 instances: 3 $Y$, 3 $N$):
     $$H(S_{\text{Strong}}) = -\left(\frac{3}{6}\log_2 \frac{3}{6} + \frac{3}{6}\log_2 \frac{3}{6}\right) = 1.0000 \text{ bits}$$

3. **Information Gain Calculation:**
   $$IG(S, \text{Wind}) = 0.9402 - \left[\frac{8}{14}(0.8113) + \frac{6}{14}(1.0000)\right] = 0.9402 - 0.8922 = 0.0480 \text{ bits}$$
Because $IG > 0$, splitting on `Wind` reduces conditional uncertainty.

---

## C. Code-to-Theory Trace

In [`src/metrics.py`](../../src/metrics.py):
* `confusion_matrix()` (lines 16–29) directly implements the contingency table tracking true vs. predicted counts $C_{ij} = |\{x \in \mathcal{D} : y = i, \hat{y} = j\}|$.
* `precision_recall_f1_per_class()` (lines 32–61) maps mathematical true positives ($TP_c = C_{cc}$), false positives ($FP_c = \sum_{i \neq c} C_{ic}$), and false negatives ($FN_c = \sum_{j \neq c} C_{cj}$) to per-class harmonic mean $F1_c$.
* `macro_f1()` (lines 64–70) averages per-class metrics uniformly, matching the course primary evaluation specification.

---

## D. One Controlled Experiment: Release Baseline

Following the R0 protocol on the UCI HAR dataset (561 features, 6 classes, subject-aware split: 7,352 train / 2,947 test), we evaluated two naive baselines in [`experiments/part1_pre_midterm/w05_baseline.py`](../../experiments/part1_pre_midterm/w05_baseline.py):

| Baseline Strategy | Test Accuracy | Test Macro-F1 | Primary Observation |
|---|---|---|---|
| **Majority-Class** (Always predict `LAYING`) | 18.22% | **0.0442** | Fails completely on 5 of 6 classes ($F1_k = 0$) |
| **Stratified Random** (Sample from class prior) | 16.59% | **0.1388** | Uniform chance level across all classes |

The experimental outputs logged in [`results/metrics.csv`](../../results/metrics.csv) establish the empirical floor: any meaningful learning model must comfortably surpass Macro-F1 $= 0.14$.

---

## E. Failure / Misconception

* **Misconception:** Maximizing training accuracy on a decision tree guarantees robust decision boundaries.
* **Reality:** Decision trees are prone to extreme variance. Without regularization (depth limits or pruning), a tree partitions feature space until leaf instances equal 1, effectively interpolating training noise and producing high generalization error.
* **Evaluation Pitfall:** Assuming high classification accuracy implies balanced performance. On skewed classes, predicting solely the dominant class yields high nominal accuracy but catastrophic Macro-F1.

---

## F. Written-Exam Capsule

> When model complexity increases, inductive bias relaxes, lowering training error while heightening sensitivity to stochastic variations in training samples (variance). The total generalization risk reaches its minimum at the point where marginal bias reduction equals marginal variance expansion. Decision trees navigate this through greedy recursive information gain splits; without early stopping (pre-pruning) or validation-guided subtree replacement (post-pruning), variance dominates, resulting in catastrophic overfitting.

---

## G. Reflection

* **Gained Competence:** Clear mathematical grounding of impurity metrics and their exact computational translation to classification trees. Deep appreciation for why Macro-F1 is mandatory over overall accuracy on multi-class sensor telemetry.
* **Remaining Uncertainty:** The quantitative difference between information gain and gain ratio when feature branching factor varies significantly.
* **Next Target:** Implementing a clean binary and multi-class Perceptron from scratch in W05, followed by the Delta rule and multi-layer perceptron backpropagation.

---

## H. Inquiry Trail

* **Question Investigated:** Minimum written drill requirements for pre-release W01–W04 baseline diagnostic.
* **Pre-AI Artifact:** Created handwritten draft [`exercises/release-baseline-w01-w04.pdf`](../../exercises/release-baseline-w01-w04.pdf) committed under hash `c521df4`.
* **Assistance Summary:** Formulated structured section breakdown and clarified syllabus alignment for Sections 7.1 and 8 of the course specification document.
* **Verification Sources:** Tom Mitchell (1997) *Machine Learning*, Chapters 1–3; Course Syllabus HK261.
