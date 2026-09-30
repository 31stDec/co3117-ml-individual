# Homework 3: Neural Networks: Representation, Learning Dynamics, and Diagnostics (Revised v2.0)

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw3`  
**Points:** 10 points (Activities 1-4: 1.5-2.5 pts, Activity 5: 2.5 pts)  
**Scope:** Chapter 3  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Linear units & Perceptron mechanics: Weight update step, decision boundary margin, and XOR non-separability.
- Activation function dynamics: Comparative trade-offs (Sigmoid, Tanh, ReLU) and vanishing gradients.
- Hand-derived Backpropagation: Tracing a forward pass, output error signal, and weight update signs.
- Learning diagnostics: Identifying underfitting, healthy convergence, and overfitting from loss curves.
- Assignment Bridge: Controlled capacity experiment (hidden width / layers / regularizer) on project dataset.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Activity 1: Perceptron step execution, linear decision boundary, and geometric proof of XOR non-separability.
2. Activity 2: Activation comparative analysis (Sigmoid, Tanh, ReLU: range, zero-centering, saturation, vanishing gradients).
3. Activity 3: Tiny backpropagation trace (x=1, hidden w=0.6, output v=-0.7, t=0): forward pass, error signals, update signs.
4. Activity 4: Diagnostic training/validation loss curves (underfitting, healthy convergence, overfitting) and interventions.
5. Activity 5 (Assignment Bridge): Controlled capacity experiment on project dataset (differing in 1 factor, MODEL_LOG reflection).

*The complete official problem sheet is provided in this folder: [`homework/hw3-problems.pdf`](hw3-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten First Attempt:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW3** at the top of page 1.
3. **Assignment Bridge:** Complete Activity 5 using your own longitudinal-assignment dataset and use case.
4. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
5. **Placement:** Name your scan exactly `homework/hw3-submission.pdf`.
6. **Validation & Tagging:**
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

## 🔍 Self-Correction & Verification Protocol
No official worked solutions will be released. After completing your handwritten submission, you are strongly encouraged to verify your calculations using course slides, study discussions with peers, and AI assistants (log your prompt strategies in `AI_USE.md`).

If you identify misconceptions or calculation errors, commit `homework/hw3-corrections.md` detailing the mistake, why it occurred, and the correct reasoning with citations. The corrections file must be committed after your original submission. Never replace, move, or delete your original submission scan or tag.
