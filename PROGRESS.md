# PROGRESS

One row per Course Week. Post and Drill are relative links; First evidence and Revision commit are
commit hashes (at least 7 characters). `python tools/check.py` verifies every row up to the current week.

| Period | Topic | Post | Drill | First evidence | Revision commit | Tag | Status |
|---|---|---|---|---|---|---|---|
| W01-W04 | PRE-RELEASE catch-up (Foundations, Decision Tree) | [catch-up](docs/pre-release/PRE_RELEASE_CATCHUP.md) | [release baseline](exercises/release-baseline-w01-w04.pdf) | c521df4 | ccadbaf | release-baseline | PRE-RELEASE |
| W05 | Perceptron, Delta Rule & Neural Networks (ANN) + onboarding | [post](docs/weekly/w05-perceptron-ann.md) | [drill](exercises/w05-first-attempt.pdf) | 48b97e7 | 3e741d5 | w05 | DONE |
| W06 | Bayesian learning / Naive Bayes | [post](docs/weekly/w06-naive-bayes.md) | [drill](exercises/w06-first-attempt.pdf) | daa90c0 | 583d6bf | w06 | DONE |
| W07 | Genetic Algorithms & Bayesian Networks (TAN), Part I closeout | [post](docs/weekly/w07-genetic-algorithms.md) | [drill](exercises/w07-first-attempt.pdf), [corrections](exercises/w07-corrections.md) | fcfbdc8 | f44ef9e | w07, part1-final | ACTIVE |
| W08 | MIDTERM (16 Oct) |  |  | n/a | n/a | w08-midterm | MIDTERM |
| W09 | HMM / sequence modelling |  |  |  |  | w09 |  |
| W10 | SVM: maximum and soft margin |  |  |  |  | w10 |  |
| W11 | Kernel SVM, cross-model comparison |  |  |  |  | w11 |  |
| W12 | PCA, curse of dimensionality |  |  |  |  | w12 |  |
| W13 | LDA, feature engineering |  |  |  |  | w13 |  |
| W14 | Bagging, boosting, AdaBoost |  |  |  |  | w14 |  |
| W15 | Generative vs discriminative, Logistic/MaxEnt, CRF, synthesis |  |  |  |  | w15, part2-final |  |

Status values: PRE-RELEASE (W01-W04), ACTIVE (current week), DONE (tag pushed), MIDTERM (W08).
W07 update (2026-10-14): scan committed at fcfbdc8 before corrections and practice revision f44ef9e. The drill used AI-provided questions and answers before handwriting, as disclosed in AI_USE.md. No independent pre-AI W07 attempt is claimed. The Part I report PDF is now present and reviewed; the handwritten A4 PDF is an interim revision with a preparation note in SUBMISSION_PART1.md. Local annotated tags `w07` and `part1-final` identify the submission package; push is pending. W07 remains ACTIVE until its tag is pushed.
W08: link the compact entry and the timed rehearsal if you have them (for example `docs/weekly/w08-midterm.md`,
`exercises/w08-rehearsal.pdf`); the two-state commit rule does not apply.
