# SUBMISSION PART 1

- Tag: `part1-final`
- Deadline: 14 October 2026, 23:59 (48 hours before midterm)
- Commit (full hash):
- Report: report/part1_summary.pdf
- Two-A4 exam sheet: exam/a4-notes-part1-midterm.pdf
- Reproduce: exact commands, environment file, seeds
- Release-day baseline and catch-up: exercises/release-baseline-w01-w04.pdf, docs/pre-release/PRE_RELEASE_CATCHUP.md
- Frozen protocol: protocol.yaml; weekly posts W05-W07 and drills linked from PROGRESS.md
- Known limitations and errata (corrections found after 14 Oct go to Part II, never into Part I history):

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
- [ ] **Decision Tree** (Depth B — catch-up): `src/from_scratch/tree_split.py`, pruning/stopping comparison
- [x] **Perceptron / Delta Rule** (Depth A): `src/from_scratch/perceptron.py`, OvR wrapper, AND/XOR sanity checks
- [ ] **MLP / Backpropagation** (Depth A): `src/from_scratch/mlp.py`, forward/backward pass, numerical gradient check
- [ ] **Bayesian / Naive Bayes** (Depth A): `src/from_scratch/naive_bayes.py`, smoothing, assumption analysis
- [ ] **Genetic Algorithm** (Depth B): representation, fitness, selection, crossover, mutation
- [ ] **Bayesian Network / TAN** (Part I Ch.6): factorization, d-separation, conditional independence

### Written-Exam Portfolio
- [x] W05 handwritten first-attempt drill (`exercises/w05-first-attempt.pdf`)
- [x] W05 corrections with citations (`exercises/w05-corrections.md`)
- [ ] W06 handwritten drill + corrections
- [ ] W07 handwritten drill + corrections
- [ ] `exam/a4-notes-part1-midterm.pdf` — Living two-A4 exam sheet (Part I version)

### Final Deliverables
- [ ] `report/part1_summary.pdf` — Part I summary report
- [ ] `MODEL_LOG.md` updated for all Part I models
- [ ] Common experimental protocol stable
- [ ] Tag `part1-final` created and pushed before 14 Oct 2026, 23:59
