# Homework 2: Decision Trees: Impurity, Splits, Pruning, and Robust Use (Revised v2.0)

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw2`  
**Points:** 10 points (five activities; the points of each activity are printed on the problem sheet)  
**Scope:** Chapter 2  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Information-theoretic intuition: Qualitative ranking of node impurities followed by exact entropy calculations.
- Splitting criteria: Evaluating candidate splits using weighted Gini impurity under CART principles.
- Continuous attribute handling: Searching candidate cut points on numeric ranges without fixed exam templates.
- Generalization & Robustness: Validation-based Reduced-Error Pruning and missing value mitigation strategies.
- Assignment Bridge: Conducting an empirical Tree Audit on the student's course project dataset.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Activity 1: Impurity intuition before calculation: ranking nodes (12,0), (9,3), (6,6) and entropy verification.
2. Activity 2: Candidate split comparison with Gini: parent (11P, 7N) splitting into S1 vs S2, CART split preference.
3. Activity 3: Continuous attributes as a search problem: numeric values (4..29), candidate cut points, impurity scoring.
4. Activity 4: Reduced-error pruning decision on validation data, strategies for missing values at test splits.
5. Activity 5 (Assignment Bridge): Tree Audit on course project dataset (depth, leaves, top 3 features, complexity experiment).

*The complete official problem sheet is provided in this folder: [`homework/hw2-problems.pdf`](hw2-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten First Attempt:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW2** at the top of page 1.
3. **Assignment Bridge:** Complete Activity 5 using your own longitudinal-assignment dataset and use case.
4. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
5. **Placement:** Name your scan exactly `homework/hw2-submission.pdf`.
6. **Validation & Tagging:**
   ```bash
   git add homework/hw2-submission.pdf
   git commit -m "[W05][homework] HW2 submission"
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

## 🔍 Self-Correction & Verification Protocol
No official worked solutions will be released. After completing your handwritten submission, you are strongly encouraged to verify your calculations using course slides, study discussions with peers, and AI assistants (log your prompt strategies in `AI_USE.md`).

If you identify misconceptions or calculation errors, commit `homework/hw2-corrections.md` detailing the mistake, why it occurred, and the correct reasoning with citations. The corrections file must be committed after your original submission. Never replace, move, or delete your original submission scan or tag.
