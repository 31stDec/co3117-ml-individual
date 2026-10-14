# W07 - Genetic Algorithms, Bayesian Networks and TAN

> **Course Week:** 7 | **Updated:** 2026-10-14
> **Depth:** GA: B; BN/TAN: C suitability study
> **Evidence:** [drill](../../exercises/w07-first-attempt.pdf), [corrections](../../exercises/w07-corrections.md), [GA experiment](../../experiments/part1_pre_midterm/w07_ga.py), [results](../../results/PART1_PRACTICE_RESULTS.md)

## A. Concept Capsule

A genetic algorithm searches a population of encoded candidates using fitness, parent selection, crossover and mutation. Our binary chromosome selects among 12 HAR features. Tournament selection supports a penalized fitness even when it is nonpositive. One unchanged elite preserves the current best, while mutation introduces alternatives; neither guarantees global optimality.

A Bayesian network factors a joint distribution according to a directed acyclic graph. Naive Bayes assumes feature independence conditional on the class. TAN retains the class parent and permits one additional feature parent per node in a tree, representing some conditional dependencies.

## B. Worked Example

For masks `101100` and `111110` with validation accuracies 0.86 and 0.89, fitness `accuracy - 0.02 * count` gives 0.80 and 0.79. Crossover after bit three yields `101110` and `111100`; flipping bit six of the first child yields `101111`. The implemented experiment instead uses Macro-F1 and a normalized count penalty to match the frozen primary metric.

For A -> C <- B, C -> D, B -> E:

$$P(A,B,C,D,E)=P(A)P(B)P(C\mid A,B)P(D\mid C)P(E\mid B).$$

Binary nodes require 1+1+4+2+2=10 independent parameters. The unobserved collider C blocks A--B; conditioning on C or descendant D activates that path. Conditioning on C blocks the chain A -> C -> D. Enumeration with specified illustrative CPTs verifies these cases; it is not a second benchmark dataset.

## C. Code-to-Theory Trace

In `src/reference_adapters/genetic_algorithm.py`, `_initialize` encodes masks, `_select` performs tournament selection, `_crossover` exchanges suffixes and `_mutate` flips bits. `run` caches fitness, repairs empty masks and records Hamming diversity. Exact equation/function locators and saved MIT reference hashes are in [DISSECTION](../reference/DISSECTION.md).

`w07_ga.py` ranks features using inner-fit ANOVA, then evaluates fitness on separate inner-validation subjects. GaussianNB is refit with the chosen mask on outer-fit subjects before outer validation. `w07_bn.py` checks joint factorization and within-class HAR correlations.

## D. Controlled Experiment

Fixed outer split: 15 fit subjects and six validation subjects. GA fitness uses an inner 11/4 subject split. Population 16, 12 generations, mutation 1/12, seeds 42, 0, 1; compare zero versus one elite with paired initial populations.

| Seed | Elites | Inner fitness | Features | Outer validation F1 |
|---|---:|---:|---:|---:|
| 42 | 0 / 1 | 0.674613 | 7 | 0.731027 |
| 0 | 0 / 1 | 0.674613 | 7 | 0.731027 |
| 1 | 0 | 0.674613 | 7 | 0.731027 |
| 1 | 1 | 0.672145 | 8 | 0.740619 |

All 12 candidates give outer F1 0.740683. GA compresses features but demonstrates no gain over that baseline. No final test data is opened. Histories, confusion matrices and diversity curves are saved.

## E. Failure / Misconception

Elitism preserves generation-best fitness but can still find a worse inner optimum within a fixed budget. Seed 1 also shows that inner fitness ordering need not transfer to outer subjects. HAR magnitude mean and energy correlate 0.868--0.988 within classes, challenging NB independence; this alone does not prove TAN improves predictions.

## F. Written-Exam Capsule

A GA balances selection with stochastic variation. Its encoding and fitness define what it can optimize. Bayesian networks express local conditional factorization; collider evidence can create dependence. TAN relaxes NB independence but requires additional conditional parameters. Schema survival arguments do not guarantee global convergence.

## G. Experiment Reflection

The results support feature compression, not a reliable performance gain. Three seeds on one holdout cannot establish population uncertainty. Continuous TAN is studied structurally; no fitted TAN accuracy is claimed. Final PDFs remain deferred.

## H. Inquiry Trail

Codex supplied the worked drill and assisted these dated implementations and reference dissection. Scan commit `fcfbdc8` preserves the handwritten copy before corrections. Existing snapshot `5422f3a` predates this work, but is not an independent W07 code attempt. Verification uses saved reference source, numerical tests, BN enumeration and measured HAR results. AI_USE records assistance; independent closed-book reproduction is pending.
