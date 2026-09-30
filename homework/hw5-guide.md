# Homework 5: Genetic Algorithms: Representation, Search, and Experimental Design (Revised v2.0)

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 23:59 PM Wednesday 14/10/2026 (UTC+7)  
**Tag name:** `hw5`  
**Points:** 10 points (Activities 1-4: 1.5-2.0 pts, Activity 5: 2.5 pts)  
**Scope:** Chapter 5  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Chromosome encoding design: Representing combinatorial search (feature selection) and handling constraint violations.
- Fitness formulation under trade-offs: Balancing predictive performance against complexity penalties.
- Genetic operators & representations: Applying mutation/crossover on bit strings and analyzing permutation constraints (TSP).
- Search dynamics & diagnostics: Diagnosing premature convergence and tuning exploration vs exploitation mechanisms.
- Assignment Bridge: Bounded search experiment on course project dataset across multiple independent random seeds.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Activity 1: Chromosome representation design for 12 candidate feature selection, decoding, handling invalid states.
2. Activity 2: Multi-objective fitness function design (accuracy reward vs feature count penalty), trade-off evaluation.
3. Activity 3: Search operators on bit strings (mutation and crossover), why bitwise operators fail on permutation problems (TSP).
4. Activity 4: Diagnosing GA runs (premature convergence, diversity collapse, stagnation), exploration vs exploitation tuning.
5. Activity 5 (Assignment Bridge): Bounded search experiment on course project dataset (feature selection across 3 seeds).

*The complete official problem sheet is provided in this folder: [`homework/hw5-problems.pdf`](hw5-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten First Attempt:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW5** at the top of page 1.
3. **Assignment Bridge:** Complete Activity 5 using your own longitudinal-assignment dataset and use case.
4. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
5. **Placement:** Name your scan exactly `homework/hw5-submission.pdf`.
6. **Validation & Tagging:**
   ```bash
   git add homework/hw5-submission.pdf
   git commit -m "[W06][homework] HW5 submission"
   python tools/check.py --checkpoint hw5
   git tag hw5
   git push origin main hw5
   ```
   *(Always push tags **one at a time**; never use `git push --tags`).*

---

## 📚 Recommended Literature & References
- Tom Mitchell (1997), *Machine Learning*, Chapter 9 (Genetic Algorithms).
- David E. Goldberg (1989), *Genetic Algorithms in Search, Optimization, and Machine Learning*.
- Course Lecture B06 slides.
- Course supplements on https://dontpad.com/CO3117_261.

---

## 🔍 Self-Correction & Verification Protocol
No official worked solutions will be released. After completing your handwritten submission, you are strongly encouraged to verify your calculations using course slides, study discussions with peers, and AI assistants (log your prompt strategies in `AI_USE.md`).

If you identify misconceptions or calculation errors, commit `homework/hw5-corrections.md` detailing the mistake, why it occurred, and the correct reasoning with citations. The corrections file must be committed after your original submission. Never replace, move, or delete your original submission scan or tag.
