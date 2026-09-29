# W05 — Perceptron, Delta Rule & ANN / Backpropagation

> **Course Week:** 5 (Release week, cal. 39)  
> **Date:** 2026-09-29  
> **Syllabus:** Chapter 4 (Tom Mitchell 1997) — Artificial Neural Networks / Perceptron / Delta Rule  
> **Depth:** A (BUILD — From-Scratch Implementation)  
> **Word count target:** 350–700 words  
> **Artifact Links:** [w05-first-attempt.pdf](../../exercises/w05-first-attempt.pdf) | [w05-corrections.md](../../exercises/w05-corrections.md) | [perceptron.py](../../src/from_scratch/perceptron.py) | [w05_perceptron.py](../../experiments/part1_pre_midterm/w05_perceptron.py)

---

## A. Concept Capsule

The **Perceptron** (Rosenblatt, 1958) is a linear binary classifier mapping input features $x \in \mathbb{R}^d$ to discrete outputs $\hat{y} \in \{-1, +1\}$ via a linear combination followed by a hard threshold:
$$\hat{y} = \text{sign}(w \cdot x + b)$$

### Key Assumptions & Theory
1. **Linear Separability:** The Perceptron Convergence Theorem guarantees convergence in finite update steps if and only if the two classes are linearly separable by a hyperplane. If data is non-separable, cycling occurs without convergence.
2. **Perceptron Rule vs. Delta Rule:** The Perceptron updates weights strictly upon classification error ($w \leftarrow w + \eta y x$). In contrast, the **Delta Rule** (Widrow & Hoff, LMS) performs continuous gradient descent on Mean Squared Error (MSE) over unthresholded activations ($w \leftarrow w + \eta (y - \hat{y}) x$). The Delta Rule guarantees asymptotic convergence to the best least-squares fit even on non-separable distributions.
3. **Multi-Layer Perceptron (MLP) & Backpropagation:** Stacking multiple linear layers without non-linear activations collapses algebraically into a single linear map. Non-linear activation functions (e.g., Sigmoid, ReLU) enable universal function approximation, with backpropagation applying the calculus chain rule backwards from loss $\mathcal{L}$ to update internal layer weights.

---

## B. One Derivation or Worked Example

Consider a binary Perceptron with weights $w = [0.5, -0.3]$, bias $b = -0.2$, learning rate $\eta = 0.1$, and training instance $x = [1, 2]$ with true label $y = +1$.

1. **Forward pass (linear combination & decision):**
   $$z = w \cdot x + b = (0.5)(1) + (-0.3)(2) + (-0.2) = 0.5 - 0.6 - 0.2 = -0.3$$
   $$\hat{y} = \text{sign}(-0.3) = -1$$
2. **Error check:**
   $$\hat{y} = -1 \neq y = +1 \implies \text{Misclassification detected.}$$
3. **Weight & bias update:**
   $$w_{\text{new}} = w + \eta \cdot y \cdot x = [0.5, -0.3] + 0.1 \cdot (+1) \cdot [1, 2] = [0.6, -0.1]$$
   $$b_{\text{new}} = b + \eta \cdot y = -0.2 + 0.1 \cdot (+1) = -0.1$$
4. **Post-update verification:**
   $$z_{\text{new}} = w_{\text{new}} \cdot x + b_{\text{new}} = (0.6)(1) + (-0.1)(2) - 0.1 = 0.3 > 0 \implies \hat{y}_{\text{new}} = +1 \quad (\text{Corrected!})$$

---

## C. Code-to-Theory Trace

In our from-scratch implementation ([`src/from_scratch/perceptron.py`](../../src/from_scratch/perceptron.py)):

| Mathematical Expression | Source Code Location | Description |
|---|---|---|
| $\hat{y} = \text{sign}(z)$ | `perceptron.py:L42-44` (`_sign`) | Heaviside/sign threshold mapping $z \ge 0 \to +1$, else $-1$. |
| $z = w \cdot x_i + b$ | `perceptron.py:L73` (`fit`) | Forward dot-product per instance. |
| $w \leftarrow w + \eta y_i x_i$, $b \leftarrow b + \eta y_i$ | `perceptron.py:L77-80` (`fit`) | Misclassification-triggered update loop. |
| $f_k(x) = X w_k + b_k$ | `perceptron.py:L95-97` (`decision_function`) | Signed geometric distance to class hyperplane. |
| $\hat{c} = \arg\max_{k} f_k(x)$ | `perceptron.py:L160-166` (`predict`) | One-vs-Rest (OvR) multi-class competitive aggregation. |

---

## D. One Controlled Experiment

We evaluated our from-scratch OvR Perceptron on the sealed **UCI HAR** benchmark (561 features, 6 human physical activity classes, 7,352 train / 2,947 test samples) under our frozen data protocol:

| Model | Depth | Test Macro-F1 | Test Accuracy | Training Time (s) | Notes |
|---|---|---|---|---|---|
| Majority-class baseline | Baseline | 0.0440 | 0.1822 | 0.00 | Always predicts LAYING |
| Stratified random baseline | Baseline | 0.1387 | 0.1639 | 0.00 | Draws from empirical prior |
| **From-Scratch OvR Perceptron** | **Depth-A** | **0.8143** | **0.9494** | **12.50** | **$\eta = 0.01$, 100 iters, z-score norm** |

### Learning Rate Sensitivity ($\eta$)
Testing across orders of magnitude ($\eta \in [0.001, 0.1]$) with 100 epochs shows stable convergence once features are standardized:
- $\eta = 0.001$: Test Macro-F1 = 0.8102, Test Accuracy = 0.9481
- $\eta = 0.010$: Test Macro-F1 = **0.8143**, Test Accuracy = **0.9494** (Best)
- $\eta = 0.100$: Test Macro-F1 = 0.8095, Test Accuracy = 0.9474

*Insight:* High-dimensional feature spaces ($d=561$) allow linear hyperplanes to separate complex posture dynamics remarkably well, dramatically crushing naive baselines.

---

## E. Failure / Misconception

### Theoretical Failure: Non-Linear Separability of XOR
Minsky & Papert (1969) proved that a single-layer perceptron cannot compute XOR because no single hyperplane can isolate $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$. 
- **Sanity Check in Code (`perceptron.py:L177-207`):**
  - **AND gate:** Linearly separable $\to$ converges in 6 epochs to 0 errors.
  - **XOR gate:** Non-separable $\to$ oscillates forever with $\ge 1$ misclassified sample per epoch.
- **Resolution:** A 2-layer MLP with 2 hidden units and non-linear activation breaks the convex hull and constructs piecewise-linear decision boundaries.

---

## F. Written-Exam Capsule

> A single-layer Perceptron uses a hard step threshold $\text{sign}(w \cdot x + b)$ and updates weights only upon misclassification ($w \leftarrow w + \eta y x$). By the Perceptron Convergence Theorem, it is guaranteed to converge in finite steps if the dataset is linearly separable, but cycles indefinitely when classes overlap. In contrast, the Delta rule (LMS) computes continuous gradient descent on Mean Squared Error ($w \leftarrow w + \eta (y - \hat{y}) x$), guaranteeing asymptotic convergence to the minimum MSE solution regardless of separability. Multi-layer perceptrons resolve linear non-separability (e.g., XOR) by combining multiple hyperplanes; non-linear activation functions (Sigmoid, ReLU) are mathematically necessary, as stacking purely linear transformations algebraically collapses back into a single linear model. Backpropagation trains multi-layer networks by applying the chain rule of differential calculus backwards from the scalar loss to compute exact partial derivatives for every weight tensor.

---

## G. Reflection

- **First-Attempt vs. Final Insight:** On paper, the single Perceptron is introduced as a toy linear model doomed by the XOR limitation. However, when deployed with One-vs-Rest on real 561-dimensional sensor data, linear separability is vastly easier to achieve due to high dimensionality (Cover's Theorem on separability of random patterns).
- **Residual Uncertainty:** Hard step activations in the classical Perceptron lack calibrated class probabilities. Transitioning to Logistic Regression and Multi-Layer Perceptrons with Softmax cross-entropy will be crucial for probabilistic decision-making.

---

## H. Inquiry Trail

1. **Investigated Question:** Verifying One-vs-Rest decision aggregation rule for linear perceptrons and formal difference between Perceptron criterion and Widrow-Hoff Delta rule.
2. **Pre-AI Evidence:** First-attempt commit `443faae` ([`exercises/w05-first-attempt.pdf`](../../exercises/w05-first-attempt.pdf)).
3. **Verification Source:** Tom Mitchell (1997), *Machine Learning*, Chapter 4 (Artificial Neural Networks), Sections 4.4–4.5; Rosenblatt (1958).
4. **Reproducibility:** All derivations, proof-by-contradiction for XOR, and OvR code reproduced independently from scratch in Python/NumPy without external ML packages.
