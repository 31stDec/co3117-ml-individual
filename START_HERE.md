# Start here

This is your private repository for CO3117 Machine Learning (HK261): one continuous record of the
semester. This page tells you what to read, what to do in your first hour, and when things are due.

## 1. Read, in this order

1. [docs/assignment/ASSIGNMENT.pdf](docs/assignment/ASSIGNMENT.pdf) (same text as
   [HTML](docs/assignment/ASSIGNMENT.html)): the full assignment. Section 8 lists the weekly topics and the
   written drill of each week, Section 9 the common experimental protocol, Section 11 the marking rubric,
   Section 12 the rules for AI and reference code, Appendix A the functions you must implement.
2. [README.md](README.md): setup, the weekly routine, the special tags and the rules the repository enforces.
3. [homework/README.md](homework/README.md): the five handwritten homework sets (problem sheets `homework/hw1-problems.pdf` to `hw5-problems.pdf`, plus a guide for each).
4. [docs/pre-release/PRE_RELEASE_CATCHUP.md](docs/pre-release/PRE_RELEASE_CATCHUP.md): the one-time catch-up for weeks 1 and 2.
5. [tools/course.yaml](tools/course.yaml): the machine-readable deadlines (the table below is generated from it).

## 2. Your first hour

- [ ] Accept the GitHub invitation (e-mail, or the Invitations page of the repository). It expires after 7 days.
- [ ] Clone the repository, create a virtual environment, `pip install -r requirements.txt`, then run `python tools/check.py`. Failures are expected at this stage; it must run.
- [ ] `git config user.name "..."` and `git config user.email "..."`.
- [ ] Open `protocol.yaml` and start choosing your dataset and use case (the default is UCI Human Activity Recognition). It must be filled and frozen by the `release-baseline` tag.
- [ ] Write the release-day baseline on paper (foundations and decision trees, closed book), scan it to `exercises/release-baseline-w01-w02.pdf` and commit it with today's date.
- [ ] Read the first homework sheet and the drill for W03 (Perceptron and Delta rule).
- [ ] Check that GitHub may e-mail you: Settings, Notifications, e-mail; keep "Participating and @mentions" on (section 5).

## 3. What is due and when

All times are Asia/Ho_Chi_Minh (UTC+7). The time that counts is the moment GitHub receives the tag.
The LMS cutoff overrides this table if they differ. The repositories were issued on 2 October, so the
setup gate `release-baseline` and the tags `w03` and `w04` fall on the same day as the first homework sets.

| Due | Tag | What |
| --- | --- | --- |
| Wed 07 Oct 2026 07:00 | `hw1` | Homework 1: Foundations: workflow, evaluation, generalisation |
| Wed 07 Oct 2026 07:00 | `hw2` | Homework 2: Decision trees |
| Wed 07 Oct 2026 07:00 | `hw3` | Homework 3: Artificial neural networks |
| Wed 07 Oct 2026 07:00 | `release-baseline` | R0 setup gate (not graded) |
| Wed 07 Oct 2026 07:00 | `w03` | W03: Perceptron / Delta + onboarding |
| Wed 07 Oct 2026 07:00 | `w04` | W04: ANN and backpropagation |
| Wed 07 Oct 2026 07:00 | `w05` | W05: Bayesian learning / Naive Bayes |
| Wed 14 Oct 2026 07:00 | `w06` | W06: Genetic Algorithm |
| Wed 14 Oct 2026 23:59 | `hw4` | Homework 4: Bayesian learning and Bayesian networks |
| Wed 14 Oct 2026 23:59 | `hw5` | Homework 5: Genetic algorithms |
| Wed 14 Oct 2026 23:59 | `part1-final` | Graded Part I (40 points) |
| Wed 14 Oct 2026 23:59 | `w07` | W07: Bayesian Networks / TAN, Part I closeout |
| Wed 21 Oct 2026 07:00 | `w08-midterm` | W08: MIDTERM (16 Oct): compact entry, timed rehearsal, reflection after the exam |
| Wed 28 Oct 2026 07:00 | `w09` | W09: HMM / sequence modelling |
| Wed 04 Nov 2026 07:00 | `w10` | W10: SVM: maximum and soft margin |
| Wed 11 Nov 2026 07:00 | `w11` | W11: Kernel SVM, cross-model comparison |
| Wed 18 Nov 2026 07:00 | `w12` | W12: PCA, curse of dimensionality |
| Wed 25 Nov 2026 07:00 | `w13` | W13: LDA, feature engineering |
| Wed 02 Dec 2026 07:00 | `w14` | W14: Bagging, boosting, AdaBoost |
| Wed 09 Dec 2026 07:00 | `w15` | W15: Generative vs discriminative, Logistic/MaxEnt, CRF, synthesis |
| two calendar days before the final exam (date to be announced) | `part2-final` | Graded Part II (60 points) |

Push one tag at a time, never `--tags`, and run `python tools/check.py --checkpoint <tag>` first.

## 4. The commit rhythm

- Every ordinary week keeps two honest states: your first attempt (committed before you open reference code or an AI tool) and the corrected version after study. Commit messages look like `[W05][code] own Gaussian NB, first attempt`, `[W05][review] corrected after Murphy 3.3`.
- Commit small and often; push at the end of each session. A single dump before a deadline does not show progress.
- Never force-push, rebase published commits, back-date a commit, or delete or move a tag. Late work stays visibly late.
- Do not edit `tools/`, `tests/contract/` or `.github/`.

## 5. Automatic reminders

A workflow opens an issue in this repository every Monday at 09:00 (and on Thursday if a deadline is within four days or you have not committed for four days). It is assigned to you, so GitHub notifies you on the website and by e-mail. Each reminder closes the previous one. The reminders are a convenience: the deadlines above are what counts.

## 6. Also required by the assignment, not machine-checked

- The two-A4 living exam sheet: start it now, update it every week (Part I version due with `part1-final`, Part II version with `part2-final`).
- `MODEL_LOG.md`: one block per model family, updated in the week you study it.
- `AI_USE.md`: one row per AI-assisted episode, and section H of that week's post (the protocol is in assignment Section 12).
- W08 (midterm week): a compact preparation entry and a timed rehearsal before the exam, `exam/midterm-reflection.md` after it, before the `w08-midterm` deadline in the table. No new major implementation.
- W15: the 90-minute mock final (`exam/mock-final-first-attempt.pdf`, then `exam/mock-final-corrections.md`).

## 7. Questions

Ask on the course LMS forum or write to the instructor with your student ID and the output of `python tools/check.py`.
