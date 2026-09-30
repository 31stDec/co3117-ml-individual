# Homework 2: Decision Trees: splitting criteria, pruning, ID3/C4.5

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw2`  
**Points:** 10 points (5 problems x 2 points; 0.5 per part A-D)  
**Scope:** Chapter 2  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Information-theoretic foundations: Shannon Entropy, Information Gain, Gain Ratio, and Gini Impurity.
- Tree induction algorithms: Step-by-step induction mechanics of ID3 and C4.5.
- Continuous feature handling: Midpoint candidate split evaluations.
- Overfitting control: Pre-pruning thresholding and Reduced-Error post-pruning algorithms.
- Inductive bias: Geometric properties of axis-parallel orthogonal decision boundaries.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Problem 1: Shannon entropy, conditional entropy, and Information Gain calculations.
2. Problem 2: Root and internal node selection for discrete attribute datasets (ID3).
3. Problem 3: Continuous attribute splitting and C4.5 Gain Ratio adjustments.
4. Problem 4: Validation-based Reduced Error Pruning and tree complexity penalties.
5. Problem 5: Inductive bias analysis and comparison with nearest-neighbor decision boundaries.

*The complete official problem sheet is provided in this folder: [`homework/hw2-problems.pdf`](hw2-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW2** at the top of page 1.
3. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
4. **Placement:** Name your scan exactly `homework/hw2-submission.pdf`.
5. **Validation & Tagging:**
   ```bash
   git add homework/hw2-submission.pdf
   git commit -m "[W03][homework] HW2 submission"
   python tools/check.py --checkpoint hw2
   git tag hw2
   git push origin main hw2
   ```
   *(Always push tags **one at a time**; never use `git push --tags`).*

---

## 📚 Recommended Literature & References
- Tom Mitchell (1997), *Machine Learning*, Chapter 3.
- J. Ross Quinlan (1986), *Induction of Decision Trees*, Machine Learning 1:81-106.
- Course Lecture B02 slides.
- ML-From-Scratch reference implementation: https://github.com/eriklindernoren/ML-From-Scratch.

---

## 🔍 Self-Correction Protocol
After the official solutions are released on Wednesday, you may review your submission and add `homework/hw2-corrections.md` detailing any errors, why they occurred, and the correct reasoning with citations. The corrections file must be committed after your submission. Never replace, move, or delete your original submission scan or tag.
