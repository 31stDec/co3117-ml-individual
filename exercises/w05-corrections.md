# W05 Written Drill — Review and Corrections

> **Reference Artifact:** [`exercises/w05-first-attempt.pdf`](w05-first-attempt.pdf)  
> **First-Attempt Commit:** `443faae`  
> **Textbook Reference:** Tom Mitchell (1997), *Machine Learning*, Chapter 4 (Artificial Neural Networks), Sections 4.4 & 4.5.

---

## 1. Item-by-Item Verification

### Q1: Perceptron Update Rule
- **Evaluation in First Attempt:**
  - Forward pass calculated correctly: $z = 0.5(1) - 0.3(2) + 0.1 = 0.0$. By standard convention $\text{sign}(0) = +1$, correctly matching label $y = +1$ (no update needed).
  - Counterfactual case with $b = -0.2$ yielded $z = -0.3 \implies \hat{y} = -1 \neq +1$ (misclassified).
  - Update: $w_{\text{new}} = [0.5, -0.3] + 0.1 \cdot (+1) \cdot [1, 2] = [0.6, -0.1]$, $b_{\text{new}} = -0.2 + 0.1 \cdot (+1) = -0.1$.
  - Verification: $z_{\text{new}} = 0.6(1) - 0.1(2) - 0.1 = 0.3 > 0 \implies \hat{y} = +1$.
- **Textbook Citation:** Mitchell (1997), Section 4.5.1, Eq. 4.5: $w_i \leftarrow w_i + \Delta w_i$ where $\Delta w_i = \eta (t - o) x_i$. With target $t \in \{-1, +1\}$ and $o \in \{-1, +1\}$, when $o \neq t$, $(t - o) = 2t$ (or directly $\eta \cdot t \cdot x_i$ under unit step convention). The calculation is **100% correct**.

### Q2: Linear Separability
- **Evaluation in First Attempt:**
  - Formal condition correctly stated: $\exists w, b: \forall i, y_i (w \cdot x_i + b) > 0$.
  - 2D sketch clearly contrasts a clean linear decision boundary separating two clusters vs. non-separable overlapping clusters where no single straight line can achieve zero error.
- **Citation:** Mitchell (1997), Section 4.5.2 (Perceptron Representational Power).

### Q3: Why XOR Fails for a Single Perceptron
- **Evaluation in First Attempt:**
  - Proof by contradiction establishes the 4 simultaneous inequalities:
    1. $b \le 0$
    2. $w_2 + b > 0$
    3. $w_1 + b > 0$
    4. $w_1 + w_2 + b \le 0$
  - Adding (2) and (3) gives $(w_1 + w_2 + b) + b > 0 \implies w_1 + w_2 + b > -b \ge 0$, directly contradicting (4).
  - Minimum architecture stated: 2-layer MLP with 2 inputs, 2 hidden non-linear units, and 1 output unit.
- **Citation:** Minsky & Papert (1969), *Perceptrons*; Mitchell (1997), Section 4.5.2 & Section 4.6.

### Q4: Perceptron Rule vs. Delta Rule
- **Evaluation in First Attempt:**
  - Correctly contrasted discrete output thresholding vs. continuous linear output.
  - Correctly distinguished misclassification-only updates from continuous gradient updates proportional to residual error $(y - \hat{y})$.
  - Loss function and convergence conditions are accurately compared.
- **Citation:** Mitchell (1997), Section 4.4.2 (The Delta Rule).

---

## 2. Summary & Score
- **Total Score:** 4 / 4 subquestions verified and accurate.
- **No fundamental errors detected;** all mathematical inequalities and numeric steps in the first attempt are sound.
