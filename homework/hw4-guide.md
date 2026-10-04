# Homework 4: Probabilistic Learning and Graphical Reasoning (Revised v2.0)

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 23:59 PM Wednesday 14/10/2026 (UTC+7)  
**Tag name:** `hw4`  
**Points:** 10 points (five activities; the points of each activity are printed on the problem sheet)  
**Scope:** Chapters 4 & 6 (Part I)  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Likelihood as evidence: Interpreting likelihood functions for parameter plausibility without rote derivation.
- Full Naive Bayes by hand: Calculating prior and conditional probabilities with Laplace add-one smoothing.
- Graphical models & d-separation: Identifying active vs blocked paths and colliders/v-structures on a DAG.
- Reasoning modes & explaining away: Distinguishing predictive vs diagnostic reasoning, common-effect conditioning.
- Assignment Bridge: Probabilistic model check (Naive Bayes baseline on project dataset, assumptions, failure case).

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Activity 1: Likelihood as evidence: biased coin 9 heads in 12 tosses, likelihood curve shaping plausible parameters.
2. Activity 2: Full Naive Bayes by hand on 8-sample binary table with Laplace add-one smoothing, conditional independence.
3. Activity 3: Bayesian Network d-separation on 5-node DAG (A->C<-B->E, C->D), blocked vs active paths under 4 conditioning sets.
4. Activity 4: Predictive vs diagnostic reasoning, explaining away effect / Berkson's paradox on common effects.
5. Activity 5 (Assignment Bridge): Probabilistic model check (Naive Bayes on project dataset, assumptions, MODEL_LOG note).

*The complete official problem sheet is provided in this folder: [`homework/hw4-problems.pdf`](hw4-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten First Attempt:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW4** at the top of page 1.
3. **Assignment Bridge:** Complete Activity 5 using your own longitudinal-assignment dataset and use case.
4. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
5. **Placement:** Name your scan exactly `homework/hw4-submission.pdf`.
6. **Validation & Tagging:**
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

## 🔍 Self-Correction & Verification Protocol
No official worked solutions will be released. After completing your handwritten submission, you are strongly encouraged to verify your calculations using course slides, study discussions with peers, and AI assistants (log your prompt strategies in `AI_USE.md`).

If you identify misconceptions or calculation errors, commit `homework/hw4-corrections.md` detailing the mistake, why it occurred, and the correct reasoning with citations. The corrections file must be committed after your original submission. Never replace, move, or delete your original submission scan or tag.
