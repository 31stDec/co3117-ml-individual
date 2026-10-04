# CO3117 - Machine Learning | Individual Longitudinal Assignment

## One Dataset, One Use Case, Many Models

**Student:** Phạm Minh Trí - 2353235  
**Semester:** HK261 (2026–2027)  
**Faculty:** Computer Science and Engineering, HCMUT, VNU-HCM  
**Official Repository:** `HCMUT-CO3117-261/ml-2353235`

---

## 🎯 Use Case

**Dataset:** UCI Human Activity Recognition Using Smartphones  
**Source:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones  
**Prediction target:** Predict a person's current physical activity from smartphone inertial measurements  
**Primary metric:** Macro-F1  
**Secondary metrics:** Accuracy, Confusion Matrix

## 📊 Data Protocol (Frozen at R0)

| Item | Value |
|------|-------|
| Dataset | UCI HAR Using Smartphones v1.0 |
| Target | Activity label (6 classes) |
| Split | Subject-aware split (no entity leakage) |
| Primary metric | Macro-F1 |
| Random seed | 42 |
| Test set | Sealed — used only for final comparison |

## 🚀 Quick Start

```bash
# Setup environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run self-check
python tools/check.py

# Run experiments
python experiments/part1_pre_midterm/w05_perceptron.py
```

---

## Course Guidelines & Workflow

This repository is your single, continuous record for the whole semester (assignment Section 5).
It lives in the course organisation; you have write access and the instructor has admin access.
Nobody else in the class can see it.

> **Read [START_HERE.md](START_HERE.md) for onboarding details.** The full assignment text is in
> [docs/assignment/ASSIGNMENT.pdf](docs/assignment/ASSIGNMENT.pdf) (also as
> [HTML](docs/assignment/ASSIGNMENT.html)); deadlines are in `START_HERE.md` and `tools/course.yaml`.

### Weekly routine (from W05)

1. Written drill, closed book: scan it to `exercises/wNN-first-attempt.pdf` (or .jpg/.png) and commit it
   immediately: `[W05][drill] first attempt`.
2. First attempt at the code for Depth-A components in `src/from_scratch/`: commit it before you open any
   reference repository or AI tool: `[W05][code] own Gaussian NB, first attempt`.
3. Study, fix, benchmark, write `exercises/wNN-corrections.md` and the weekly post
   `docs/weekly/wNN-<topic>.md` from `docs/weekly/_TEMPLATE.md`.
4. Fill the week's row in `PROGRESS.md` (post and drill links, first-attempt hash, revision hash).
5. `python tools/check.py --checkpoint wNN` until it passes, then tag and push:

```bash
git tag wNN
git push course wNN
```

Pushing the tag runs the same check in the Actions tab. Push checkpoint tags **one at a time**,
never with `--tags`; GitHub does not run the check when more than three tags arrive in one push.
Deadlines for each tag are in `tools/course.yaml`.

### Checkpoint Tags

| Tag | What it marks | Deadline |
| --- | --- | --- |
| `release-baseline` | R0 setup gate: skeleton, catch-up post, release-day baseline scan `exercises/release-baseline-w01-w04.pdf`, frozen `protocol.yaml` | `tools/course.yaml` (07 Oct 07:00) |
| `w05` | W05 Perceptron, Delta Rule & ANN + onboarding | `tools/course.yaml` (07 Oct 07:00) |
| `hw1` to `hw5` | One handwritten homework set each (under `homework/`) | `tools/course.yaml` |
| `w08-midterm` | Midterm week: compact entry, timed rehearsal scan, then `exam/midterm-reflection.md` | `tools/course.yaml` |
| `part1-final` | Graded Part I package (`SUBMISSION_PART1.md`, `report/part1_summary.pdf`, `exam/a4-notes-part1-midterm.pdf`) | 14 Oct 2026 23:59 |
| `part2-final` | Graded Part II package (adds the mock final and `exam/a4-notes-part2-final.pdf`) | two calendar days before final exam |

### Handwritten Homework Sets (Part I)

Five homework sets, one per Part I topic. The problem sheets are in `homework/` and on the course LMS.
They are formative handwritten practice, separate from the weekly drill, and reviewed under the Written-exam portfolio criterion.

1. Write answers by hand (paper or stylus), with name, student ID and set number on page 1.
2. Scan all pages into one PDF of at most 5 MB: `homework/hwN-submission.pdf`.
3. Commit it (`[W05][homework] HW1 submission`), run `python tools/check.py --checkpoint hwN`, then:

```bash
git tag hwN
git push course hwN
```

### Important Integrity Rules
- The history of `main` and every checkpoint tag are permanent. Do not force-push, rebase published commits, delete or move a tag.
- Do not edit `tools/`, `tests/contract/` or `.github/`. They are compared with the released version.
- Never commit raw data, archives, model binaries or secrets (see `.gitignore` and `data/README.md`).
