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
| W10 | SVM: maximum and soft margin | [post](docs/weekly/w10-svm.md) | [drill](exercises/w10-first-attempt.pdf), [corrections](exercises/w10-corrections.md) | 358a662 | 1947792 | w10 | DONE |
| W11 | Kernel SVM, cross-model comparison | [post](docs/weekly/w11-kernel-svm.md) | [drill](exercises/w11-first-attempt.pdf), [corrections](exercises/w11-corrections.md) | b01f02d | d0a8c71 | w11 | DONE |
| W12 | PCA, curse of dimensionality | [post](docs/weekly/w12-pca.md) | [drill](exercises/w12-first-attempt.pdf), [corrections](exercises/w12-corrections.md) | 94f024a | 5b70e8c | w12 | DONE |
| W13 | LDA, feature engineering | [post](docs/weekly/w13-lda.md) | [drill](exercises/w13-first-attempt.pdf), [corrections](exercises/w13-corrections.md) | 4c5fa6a | dae6451 | w13 | DONE |
| W14 | Bagging, boosting, AdaBoost |  |  |  |  | w14 |  |
| W15 | Generative vs discriminative, Logistic/MaxEnt, CRF, synthesis |  |  |  |  | w15, part2-final |  |

Status values: PRE-RELEASE (W01-W04), ACTIVE (current week), DONE (tag pushed), MIDTERM (W08).

W13 update (2026-11-24): Scan `4c5fa6a` is committed. LDA experiments, results, corrections, weekly post, and assistance record are committed at `dae6451`. Tag `w13` has been created. Local checkpoint validation passed 81/81 checks. A4 updates remain deferred.

W12 update (2026-11-17): Scan `94f024a` is committed. PCA code, reference checks, HAR experiments, corrections, weekly post, and assistance record are committed at `5b70e8c`. Tag `w12` at `1e9310d` has been pushed to the course repository. Local checkpoint validation passed 74/74 checks. A4 remains deferred.

W11 update (2026-11-10): Scan `b01f02d` and revision `d0a8c71` are committed. The revision includes corrections, reference study, HAR results, weekly post, and assistance record. Tag `w11` at `cdc7067` has been pushed to the course repository. Local checkpoint validation passed 66/66 checks. A4 remains deferred.

W10 update (2026-11-03): Scan `358a662` is committed. Corrections, SVM reference study, HAR experiments, results, and the weekly post are committed at `1947792`. The w10 tag at `69b9e2f` has been pushed to the course repository. A4 updates remain deferred.

W09 update (2026-10-27): Scan `969096b`, HMM code `49f5595`, and review/experiments `c833449` are committed. The w09 checkpoint is being validated for submission. A4 updates are deferred to final-exam preparation; the wider assistance record remains pending.

W07 update (2026-10-14): scan committed at c6d1880 before corrections and practice revision 6f2ec9d. Assistance and verification are recorded in AI_USE.md. The Part I report PDF is now present and reviewed; the handwritten A4 PDF is an interim revision with a preparation note in SUBMISSION_PART1.md. Local annotated tags `w07` and `part1-final` identify the submission package; push is pending. W07 remains ACTIVE until its tag is pushed.
W08 update (2026-10-19): added the post-exam reflection and the cheat sheet used on 16 October as post-exam evidence. The compact entry links both versions of the notes and records the evidence limitations. No separate timed-rehearsal scan is linked in this repository. The two-state commit rule does not apply to W08.
