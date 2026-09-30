# Homework 1: Foundations: workflow, evaluation, generalisation

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw1`  
**Points:** 10 points (5 problems x 2 points; 0.5 per part A-D)  
**Scope:** Chapter 1  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Rigorous mathematical understanding of performance metrics: Confusion Matrix, Accuracy, Precision, Recall, F1-score, ROC curves, and Precision-Recall tradeoffs.
- Validation methodologies: K-fold cross-validation, stratified splits, data leakage prevention, and hyperparameter optimization protocols.
- Generalisation bounds: Analytical decomposition of Bias-Variance tradeoff, identifying overfitting/underfitting regimes.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Problem 1: Experimental setup, train/val/test split design, and data leakage identification.
2. Problem 2: Confusion matrix derivation, Precision-Recall tradeoffs, and F1 calculations.
3. Problem 3: K-fold and Stratified Cross-Validation mechanics, variance vs bias of estimators.
4. Problem 4: Model complexity analysis, learning curves, and structural risk minimization.
5. Problem 5: Mathematical derivation of Bias-Variance decomposition for squared error loss.

*The complete official problem sheet is provided in this folder: [`homework/hw1-problems.pdf`](hw1-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW1** at the top of page 1.
3. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
4. **Placement:** Name your scan exactly `homework/hw1-submission.pdf`.
5. **Validation & Tagging:**
   ```bash
   git add homework/hw1-submission.pdf
   git commit -m "[W03][homework] HW1 submission"
   python tools/check.py --checkpoint hw1
   git tag hw1
   git push origin main hw1
   ```
   *(Always push tags **one at a time**; never use `git push --tags`).*

---

## 📚 Recommended Literature & References
- Tom Mitchell (1997), *Machine Learning*, Chapter 1.
- Hastie, Tibshirani, Friedman (2009), *The Elements of Statistical Learning*, Chapter 7 (Model Assessment and Selection).
- Course Lecture B01 & B02 slides: `Share Students/01_Tong-quan/`.
- Supplementary study materials at https://dontpad.com/CO3117_261.

---

## 🔍 Self-Correction Protocol
After the official solutions are released on Wednesday, you may review your submission and add `homework/hw1-corrections.md` detailing any errors, why they occurred, and the correct reasoning with citations. The corrections file must be committed after your submission. Never replace, move, or delete your original submission scan or tag.
