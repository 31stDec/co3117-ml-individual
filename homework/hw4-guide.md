# Homework 4: Bayesian Learning and Bayesian Networks: MAP, MLE, Naive Bayes, DAGs

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 23:59 PM Wednesday 14/10/2026 (UTC+7)  
**Tag name:** `hw4`  
**Points:** 10 points (5 problems x 2 points; 0.5 per part A-D)  
**Scope:** Chapters 4 & 6 (Part I)  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Probabilistic learning paradigms: Maximum Likelihood Estimation (MLE) vs Maximum A Posteriori (MAP).
- Generative classification: Naive Bayes conditional independence assumption, prior, and likelihood estimation.
- Zero-probability smoothing: Laplace correction and Lidstone / m-estimate smoothing.
- Bayesian Belief Networks (BBN): Directed Acyclic Graphs (DAG), joint distribution factorizations.
- Conditional independence: d-separation criteria, explaining away, Markov blanket, and Tree-Augmented Naive Bayes (TAN).

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Problem 1: Analytical derivations of MLE and MAP estimators for Bernoulli and Gaussian distributions.
2. Problem 2: Discrete Naive Bayes classification with evidence log-posterior computations.
3. Problem 3: Laplace smoothing and m-estimate smoothing impact on sparse text/categorical data.
4. Problem 4: Bayesian Network DAG factorization, Conditional Probability Tables (CPT), and joint queries.
5. Problem 5: d-separation analysis, blocked paths under observed evidence, and TAN model structure.

*The complete official problem sheet is provided in this folder: [`homework/hw4-problems.pdf`](hw4-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW4** at the top of page 1.
3. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
4. **Placement:** Name your scan exactly `homework/hw4-submission.pdf`.
5. **Validation & Tagging:**
   ```bash
   git add homework/hw4-submission.pdf
   git commit -m "[W07][homework] HW4 submission"
   python tools/check.py --checkpoint hw4
   git tag hw4
   git push origin main hw4
   ```
   *(Always push tags **one at a time**; never use `git push --tags`).*

---

## 📚 Recommended Literature & References
- Tom Mitchell (1997), *Machine Learning*, Chapter 6.
- Kevin Murphy (2022), *Probabilistic Machine Learning: An Introduction*, Chapters 3 & 4.
- UPenn CIS 262 Notes on HMM and Probabilistic Graphical Models: https://www.engineering.upenn.edu/~cis2620/notes/cis262-hmm.pdf.
- Course Lecture B05 & B07 slides.

---

## 🔍 Self-Correction Protocol
After the official solutions are released on Wednesday, you may review your submission and add `homework/hw4-corrections.md` detailing any errors, why they occurred, and the correct reasoning with citations. The corrections file must be committed after your submission. Never replace, move, or delete your original submission scan or tag.
