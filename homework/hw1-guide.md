# Homework 1: Foundations, Evaluation, and Generalisation (Revised v2.0)

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 07:00 AM Wednesday 07/10/2026 (UTC+7)  
**Tag name:** `hw1`  
**Points:** 10 points (five activities; the points of each activity are printed on the problem sheet)  
**Scope:** Chapter 1  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Rigorous framing of ML tasks: input, target, decision context, label noise, and task formulation.
- Validation methodologies under data constraints: 80/20 splits, K-fold CV, and detecting test-set leakage.
- Performance metrics & asymmetric decision costs: Accuracy, Precision, Recall, F1, and cost sensitivity.
- Diagnostic reasoning on learning behavior: Trios of train/val errors, bias-variance trade-offs vs complexity.
- Assignment Bridge: Authoring a comprehensive Protocol Card for the student's longitudinal course dataset.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Activity 1: Learning problem framing (postal envelope routing, task objectives, classification vs regression).
2. Activity 2: Evaluation under limited data (120 samples, 80/20 vs 5-fold CV, hyperparameter tuning leakage).
3. Activity 3: Metrics and decision context (confusion matrix TP=36, FN=14, FP=30, TN=170, rare-event costs).
4. Activity 4: Diagnostic model behaviour (underfitting, healthy, overfitting train/val profiles, bias-variance).
5. Activity 5 (Assignment Bridge): 1-page Protocol Card for project dataset (target, split, metric, leakage, fairness).

*The complete official problem sheet is provided in this folder: [`homework/hw1-problems.pdf`](hw1-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten First Attempt:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW1** at the top of page 1.
3. **Assignment Bridge:** Complete Activity 5 using your own longitudinal-assignment dataset and use case.
4. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
5. **Placement:** Name your scan exactly `homework/hw1-submission.pdf`.
6. **Validation & Tagging:**
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
- Hastie, Tibshirani, Friedman (2009), *The Elements of Statistical Learning*, Chapter 7 (Model Assessment).
- Course Lecture B01 & B02 slides: `Share Students/01_Tong-quan/`.
- Supplementary study materials at https://dontpad.com/CO3117_261.

---

## 🔍 Self-Correction & Verification Protocol
No official worked solutions will be released. After completing your handwritten submission, you are strongly encouraged to verify your calculations using course slides, study discussions with peers, and AI assistants (log your prompt strategies in `AI_USE.md`).

If you identify misconceptions or calculation errors, commit `homework/hw1-corrections.md` detailing the mistake, why it occurred, and the correct reasoning with citations. The corrections file must be committed after your original submission. Never replace, move, or delete your original submission scan or tag.
