# Homework 3: Artificial Neural Networks: Perceptron, MLP, Backpropagation

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw3`  
**Points:** 10 points (5 problems x 2 points; 0.5 per part A-D)  
**Scope:** Chapter 3  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Linear units & Perceptron convergence theorem: Margin geometry, delta rule, LMS gradient descent.
- Multi-Layer Perceptrons (MLP): Forward pass tensor mechanics and non-linear representations.
- Analytical Backpropagation: Deriving exact partial derivatives of loss via multi-variable chain rule.
- Activation functions: Gradient dynamics of Sigmoid, Tanh, ReLU, and vanishing/exploding gradient phenomena.
- Optimization & Regularization: Learning rate scheduling, momentum, and L2/L1 weight decay penalties.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Problem 1: Perceptron weight updates, hyperplane margin, and linearly separable bounds.
2. Problem 2: Multi-layer feedforward network forward propagation and XOR representation.
3. Problem 3: Manual step-by-step Backpropagation gradient calculation for hidden & output weights.
4. Problem 4: Activation function derivatives, saturation regions, and vanishing gradient mitigation.
5. Problem 5: Loss function dynamics (MSE vs Cross-Entropy) and L2 regularization weight updates.

*The complete official problem sheet is provided in this folder: [`homework/hw3-problems.pdf`](hw3-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW3** at the top of page 1.
3. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
4. **Placement:** Name your scan exactly `homework/hw3-submission.pdf`.
5. **Validation & Tagging:**
   ```bash
   git add homework/hw3-submission.pdf
   git commit -m "[W04][homework] HW3 submission"
   python tools/check.py --checkpoint hw3
   git tag hw3
   git push origin main hw3
   ```
   *(Always push tags **one at a time**; never use `git push --tags`).*

---

## 📚 Recommended Literature & References
- Tom Mitchell (1997), *Machine Learning*, Chapter 4.
- Goodfellow, Bengio, Courville (2016), *Deep Learning*, Chapter 6 (Deep Feedforward Networks).
- Course Lecture B03 & B04 slides.
- Reference implementations at https://dontpad.com/CO3117_261.

---

## 🔍 Self-Correction Protocol
After the official solutions are released on Wednesday, you may review your submission and add `homework/hw3-corrections.md` detailing any errors, why they occurred, and the correct reasoning with citations. The corrections file must be committed after your submission. Never replace, move, or delete your original submission scan or tag.
