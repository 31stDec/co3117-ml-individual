# Homework 5: Genetic Algorithms: representation, selection, crossover, mutation

**Course:** CO3117 Machine Learning — HK261  
**Due date:** 23:59 PM Wednesday 14/10/2026 (UTC+7)  
**Tag name:** `hw5`  
**Points:** 10 points (5 problems x 2 points; 0.5 per part A-D)  
**Scope:** Chapter 5  

---

## 🎯 Pedagogical Objectives (Mục tiêu sư phạm)
- Evolutionary computation: Formulating optimization and search problems as evolutionary processes.
- Chromosome encoding: Binary strings, permutation representations, and fitness landscape design.
- Selection dynamics: Roulette wheel fitness-proportionate selection, tournament selection, and selection pressure.
- Genetic operators: One-point, two-point, uniform crossover, bit-flip and swap mutation mechanics.
- Theoretical principles: Holland's Schema Theorem, building block hypothesis, exploration vs exploitation trade-offs.

---

## 📋 Problem Outline (Tóm tắt bài tập)
1. Problem 1: Chromosome representation, fitness scaling, and avoiding premature convergence.
2. Problem 2: Roulette wheel probabilities vs Tournament selection with varying tournament size.
3. Problem 3: Single-point and uniform crossover execution, offspring schema preservation.
4. Problem 4: Mutation rate sensitivity, genetic drift, and diversity maintenance in populations.
5. Problem 5: Holland's Schema Theorem: calculating order, defining length, and expected schema growth.

*The complete official problem sheet is provided in this folder: [`homework/hw5-problems.pdf`](hw5-problems.pdf).*

---

## ✍️ Submission Instructions
1. **Handwritten:** Solve all problems by hand on paper or with a digital stylus. Show full derivations; a bare final number earns 0 credit.
2. **Identification:** Put your **Full Name**, **Student ID**, and **HW5** at the top of page 1.
3. **Scan:** Scan all pages in order into **one single PDF file of at most 5 MB**.
4. **Placement:** Name your scan exactly `homework/hw5-submission.pdf`.
5. **Validation & Tagging:**
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

## 🔍 Self-Correction Protocol
After the official solutions are released on Wednesday, you may review your submission and add `homework/hw5-corrections.md` detailing any errors, why they occurred, and the correct reasoning with citations. The corrections file must be committed after your submission. Never replace, move, or delete your original submission scan or tag.
