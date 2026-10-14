# SUBMISSION PART 1

- Tags: `w07`, `part1-final` (local annotated checkpoints; push pending)
- Deadline: 14 October 2026, 23:59 (48 hours before midterm)
- Commit: the final package is identified by `part1-final^{commit}`; obtain its full hash with `git rev-parse "part1-final^{commit}"`. Completed practice revision `6f2ec9da576b1cb23db3c987f266d1c675f52cec`, with scan evidence `c6d1880cd4abf16d2b7ffef7399bcf876e59f2fb` preceding corrections.
- Report: report/part1_summary.pdf
- Two-A4 exam sheet: exam/a4-notes-part1-midterm.pdf — uploaded interim handwritten revision; see the submission note below.
- Reproduce: see `experiments/part1_pre_midterm/README.md`; tested package versions in `requirements-part1-lock.txt` and `results/audit_environment.json`. Subject split seed 42; paired MLP/GA seeds 42, 0, 1.
- Release-day baseline and catch-up: exercises/release-baseline-w01-w04.pdf, docs/pre-release/PRE_RELEASE_CATCHUP.md
- Frozen protocol: protocol.yaml; weekly posts W05-W07 and drills linked from PROGRESS.md
- Known limitations: historical test partition was exposed during the original Perceptron sweep; new practice uses validation only. GA searches twelve candidate features, MLP/GA use three training seeds on one subject holdout, and TAN is a structural suitability study without a fitted TAN benchmark. Dated experiments and review are AI-assisted and do not establish independent closed-book reproduction. See `results/PART1_METRIC_AUDIT.md` and `results/PART1_PRACTICE_RESULTS.md`. Corrections after the Part I freeze belong to Part II.

---

## Part I Verification Checklist

### Setup & Catch-up
- [x] Repository skeleton with required structure
- [x] PROGRESS.md with all weeks linked
- [x] AI_USE.md initialized with IBL protocol
- [x] Honest W01–W04 catch-up (`docs/pre-release/PRE_RELEASE_CATCHUP.md`)
- [x] Release-day baseline diagnostic (`exercises/release-baseline-w01-w04.pdf`)
- [x] Dataset/use-case frozen in `protocol.yaml`
- [x] Tag `release-baseline` created

### Content — Pre-Midterm Models
- [x] **Decision Tree** (Depth B — catch-up): existing split core; attributed continuous-tree adapter, stopping/leaf-size and sklearn pruning comparisons (`w05_tree.py`, `docs/reference/DISSECTION.md`)
- [x] **Perceptron / Delta Rule** (Depth A): `src/from_scratch/perceptron.py`, OvR wrapper, AND/XOR sanity checks
- [x] **MLP / Backpropagation** (Depth A core): existing NumPy forward/backward and numerical gradient checks; AI-assisted SGD wrapper and 32/64-unit capacity comparison across three seeds
- [x] **Bayesian / Naive Bayes** (Depth A core): corrected global-variance smoothing; HAR validation benchmark, sklearn agreement, probability and error analysis
- [x] **Genetic Algorithm** (Depth B): attributed feature-mask search with nested validation, tournament selection, crossover, mutation, elite ablation and three seeds
- [x] **Bayesian Network / TAN** (Part I Ch.6): factorization and d-separation enumeration; within-class HAR correlations and structural suitability study. No fitted TAN performance is claimed.

### Written-Exam Portfolio
- [x] W05 handwritten first-attempt drill (`exercises/w05-first-attempt.pdf`)
- [x] W05 corrections with citations (`exercises/w05-corrections.md`)
- [x] W06 handwritten drill + corrections
- [x] W07 handwritten worked review + corrections (AI assistance disclosed; scan committed at c6d1880 before corrections)
- [ ] `exam/a4-notes-part1-midterm.pdf` — Interim handwritten revision is present; consolidation is not yet complete. See status note below.

### Final Deliverables
- [x] `report/part1_summary.pdf` — Part I summary report; Overleaf export reviewed on 14 October 2026
- [x] `MODEL_LOG.md` updated for all Part I models and suitability studies
- [x] Dataset, target and official train/test populations unchanged; internal validation and test-exposure limitation documented
- [x] Local annotated tags `w07` and `part1-final` created for the submission package
- [ ] Push `main`, `w07` and `part1-final` before 14 Oct 2026, 23:59

## Current working-tree package

- `results/PART1_PRACTICE_RESULTS.md`: measured Tree/MLP/GA results, BN/TAN checks and interpretation.
- `results/validation_metrics.csv`: per-configuration validation results; historical test reruns remain separately in `results/metrics.csv`.
- `results/*_audit.json`: subject identities, settings, confusion matrices, loss/search histories and selected features.
- `results/figures/`: three checked figures for tree complexity, MLP capacity and GA elitism/diversity.
- `docs/experiments/PART1_PLAN.md`: settings and expected effects recorded before running.
- `docs/reference/`: exact reference snapshots, source hashes, MIT license and dissection/adaptation mapping.
- Final report: `report/part1_summary.pdf` is present; the 11-page Overleaf export has been reviewed for layout and the previously overflowing paths now wrap within the margins.
- Two-A4 sheet: current two-page handwritten revision is present; completion and further mock-test-based refinement are planned, as recorded below.

## Handwritten A4 notes - interim submission note (2026-10-14)

I started revising these handwritten notes after the mock test on Sunday, 11 October. This submitted version records my current progress; the consolidation is not yet complete because of my examination schedule. After completing two other examinations on the morning of 15 October, I plan to continue and complete the notes. I will then refine them again using the online mock test scheduled for that afternoon, before the ML midterm on 16 October.
