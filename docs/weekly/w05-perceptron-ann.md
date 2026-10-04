# W05 — Perceptron, Delta Rule & ANN / Backpropagation

> **Course Week:** 5 | **Date:** 2026-09-29  
> **Syllabus:** Chapter 4 (Tom Mitchell 1997) — Artificial Neural Networks / Perceptron  
> **Depth:** A (BUILD)  
> **Links:** [w05-first-attempt.pdf](../../exercises/w05-first-attempt.pdf) | [w05-corrections.md](../../exercises/w05-corrections.md) | [perceptron.py](../../src/from_scratch/perceptron.py)

---

## A. Concept Capsule

The **Perceptron** (Rosenblatt, 1958) maps input $x \in \mathbb{R}^d$ to $\hat{y} \in \{-1, +1\}$ via a hard threshold:
$$\hat{y} = \text{sign}(w \cdot x + b)$$

1. **Perceptron Rule vs. Delta Rule:** The Perceptron updates weights only on misclassification ($w \leftarrow w + \eta y x$). It is guaranteed to converge in finite steps if and only if data is linearly separable (Perceptron Convergence Theorem). The **Delta Rule** (Widrow-Hoff) uses continuous gradient descent on Mean Squared Error ($w \leftarrow w + \eta (y - \hat{y}) x$), converging asymptotically regardless of separability.
2. **Multi-Layer Perceptron (MLP):** Overcomes linear separability limits (e.g., XOR). Stacking linear layers collapses algebraically into a single linear map; non-linear activations (tanh, ReLU) are mathematically necessary for universal approximation.

---

## B. One Derivation or Worked Example

For Perceptron with $w = [0.5, -0.3]$, $b = -0.2$, $\eta = 0.1$, and example $x = [1, 2]$, $y = +1$:
1. **Forward pass:**
   $$z = w \cdot x + b = 0.5(1) - 0.3(2) - 0.2 = -0.3 \implies \hat{y} = -1 \neq y$$
2. **Weight update on mistake:**
   $$w_{\text{new}} = [0.5, -0.3] + 0.1(+1)[1, 2] = [0.6, -0.1]$$
   $$b_{\text{new}} = -0.2 + 0.1(+1) = -0.1$$
3. **Verification:**
   $$z_{\text{new}} = 0.6(1) - 0.1(2) - 0.1 = 0.3 > 0 \implies \hat{y}_{\text{new}} = +1 \quad (\text{Corrected})$$

---

## C. Code-to-Theory Trace

In [`src/from_scratch/perceptron.py`](../../src/from_scratch/perceptron.py):
* `perceptron_update()`: Implements $w \leftarrow w + \eta y x$ and $b \leftarrow b + \eta y$ when $y(w \cdot x + b) \le 0$.
* `Perceptron.fit()`: Iterates over training samples, calling update on misclassified instances.
* `MultiClassPerceptron.predict()`: One-vs-Rest argmax over signed decision scores $\arg\max_k (w_k \cdot x + b_k)$.

---

## D. One Controlled Experiment

On UCI HAR (561 features, 6 classes, subject-aware split), we evaluated our from-scratch OvR Perceptron:

| Model | Depth | Test Macro-F1 | Test Accuracy | Train Time (s) |
|---|---|---|---|---|
| Majority-class baseline | Baseline | 0.0440 | 0.1822 | 0.00 |
| Stratified random baseline | Baseline | 0.1387 | 0.1639 | 0.00 |
| **From-Scratch OvR Perceptron** | **Depth-A** | **0.8143** | **0.9494** | **12.50** |

High dimensionality ($d=561$) makes linear separation effective, drastically outperforming baselines.

---

## E. Failure / Misconception

* **Theoretical Failure (XOR):** Minsky & Papert (1969) proved a single hyperplane cannot isolate $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$. Code sanity checks confirm AND converges in 6 epochs, while XOR cycles indefinitely with errors $> 0$.
* **Resolution:** Multi-layer networks with non-linear hidden units form non-linear decision boundaries.

---

## F. Written-Exam Capsule

> The Perceptron updates weights strictly upon error using a step function threshold. By the Perceptron Convergence Theorem, it converges in finite steps on linearly separable data but cycles infinitely on non-separable sets. The Delta rule minimizes mean squared error via continuous gradient descent, ensuring asymptotic convergence to the best least-squares fit. Stacking multiple layers requires non-linear activations; purely linear layers algebraically collapse to a single linear transformation. Backpropagation trains multi-layer networks by propagating loss gradients backwards via the calculus chain rule.

---

## G. Reflection

* **Gained Insight:** High feature dimensions ($d=561$) substantially increase linear separability odds, explaining the strong 81.4% Macro-F1.
* **Next Target:** Implementing Gaussian Naive Bayes and handling feature variance smoothing in W06.

---

## H. Inquiry Trail

1. **Question:** Verifying OvR multi-class aggregation and Perceptron vs. Delta rule distinctions.
2. **Pre-AI Evidence:** Commit `443faae` ([`exercises/w05-first-attempt.pdf`](../../exercises/w05-first-attempt.pdf)).
3. **Verification Source:** Tom Mitchell (1997) *Machine Learning*, Chapter 4, Sections 4.4–4.5.
4. **Reproduced:** Verified XOR contradiction proof and implemented complete OvR Perceptron from scratch in NumPy.
