# PROGRESS

One row per Course Week. Post and Drill are relative links; First evidence and Revision commit are
commit hashes (at least 7 characters). `python tools/check.py` verifies every row up to the current week.

| Period | Topic | Post | Drill | First evidence | Revision commit | Tag | Status |
|---|---|---|---|---|---|---|---|
| W01-W04 | PRE-RELEASE catch-up (Foundations, Decision Tree) | [catch-up](docs/pre-release/PRE_RELEASE_CATCHUP.md) | [release baseline](exercises/release-baseline-w01-w04.pdf) | c521df4 | ccadbaf | release-baseline | PRE-RELEASE |
| W05 | Perceptron, Delta Rule & Neural Networks (ANN) + onboarding | [post](docs/weekly/w05-perceptron-ann.md) | [drill](exercises/w05-first-attempt.pdf) | 48b97e7 | 3e741d5 | w05 | DONE |
| W06 | Bayesian learning / Naive Bayes | [post](docs/weekly/w06-naive-bayes.md) | [drill](exercises/w06-first-attempt.pdf) | daa90c0 | 583d6bf | w06 | DONE |
| W07 | Genetic Algorithms & Bayesian Networks (TAN), Part I closeout | [post](docs/weekly/w07-genetic-algorithms.md) | [drill](exercises/w07-first-attempt.pdf), [corrections](exercises/w07-corrections.md) | c6d1880 | 6f2ec9d | w07, part1-final | ACTIVE |
| W08 | MIDTERM (16 Oct) | [entry](docs/weekly/w08-midterm.md), [reflection](exam/midterm-reflection.md) |  | n/a | n/a | w08-midterm | MIDTERM |
| W09 | HMM / sequence modelling | [post](docs/weekly/w09-hmm.md) | [drill](exercises/w09-first-attempt.pdf), [corrections](exercises/w09-corrections.md) | 969096b | c833449 | w09 | ACTIVE |
| W10 | SVM: maximum and soft margin |  |  |  |  | w10 |  |
| W11 | Kernel SVM, cross-model comparison |  |  |  |  | w11 |  |
| W12 | PCA, curse of dimensionality |  |  |  |  | w12 |  |
| W13 | LDA, feature engineering |  |  |  |  | w13 |  |
| W14 | Bagging, boosting, AdaBoost |  |  |  |  | w14 |  |
| W15 | Generative vs discriminative, Logistic/MaxEnt, CRF, synthesis |  |  |  |  | w15, part2-final |  |

Status values: PRE-RELEASE (W01-W04), ACTIVE (current week), DONE (tag pushed), MIDTERM (W08).

W09 update (2026-10-27): Scan `969096b`, HMM code `49f5595`, and review/experiments `c833449` are committed. The w09 checkpoint is being validated for submission. A4 updates are deferred to final-exam preparation; the wider assistance record remains pending.

W07 update (2026-10-14): scan committed at c6d1880 before corrections and practice revision 6f2ec9d. Assistance and verification are recorded in AI_USE.md. The Part I report PDF is now present and reviewed; the handwritten A4 PDF is an interim revision with a preparation note in SUBMISSION_PART1.md. Local annotated tags `w07` and `part1-final` identify the submission package; push is pending. W07 remains ACTIVE until its tag is pushed.
W08 update (2026-10-19): added the post-exam reflection and the cheat sheet used on 16 October as post-exam evidence. The compact entry links both versions of the notes and records the evidence limitations. No separate timed-rehearsal scan is linked in this repository. The two-state commit rule does not apply to W08.
