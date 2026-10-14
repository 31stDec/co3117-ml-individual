# W07 Written Drill - Review and Corrections

> **Reviewed on:** 2026-10-14
> **Artifact:** [Handwritten W07 drill](w07-first-attempt.pdf), two pages.
> **Scan commit:** `fcfbdc8b10950367b5b1edc643bbccee3747d3b7`, added before this corrections file.
> **Assistance:** The questions and worked answers were supplied in the Codex conversation before the handwritten copy. This is an AI-assisted worked review, not evidence of independent closed-book reproduction. The original scan is preserved.

## Q1. Genetic algorithm

The computations and operators are correct:

- `101100` selects three features: fitness = 0.86 - 0.02 * 3 = 0.80.
- `111110` selects five features: fitness = 0.89 - 0.02 * 5 = 0.79.
- Crossover after bit 3 produces `101110` and `111100`.
- Flipping bit 6 of the first child produces `101111`, selecting five features.
- Using validation rather than test accuracy for search correctly avoids test information influencing model selection.

No substantive correction is needed.

## Q2. Bayesian-network factorization

For the graph A -> C <- B, C -> D, B -> E, the joint factorization is correct:

$$P(A,B,C,D,E)=P(A)P(B)P(C\mid A,B)P(D\mid C)P(E\mid B).$$

Each binary node with k binary parents needs 2^k independent parameters. The total is 1 + 1 + 4 + 2 + 2 = 10. No correction is needed.

## Q3. D-separation

All four conclusions are correct:

- A and B are d-separated without evidence: C is an unobserved collider.
- Given C, the collider path is active, so the graph does not imply A independent of B.
- Given D, a descendant of C is observed, also activating the collider path.
- Given C, the chain A -> C -> D is blocked, so A and D are d-separated.

The explaining-away description is also correct. An active path means the graph does not guarantee independence; particular parameter choices can still produce independence.

## Q4. Naive Bayes versus TAN - notation correction

**Observed issue, page 2:** The NB formula uses commas in `P(X2,Y)` and `P(X3,Y)` where conditional bars are required. A comma denotes a joint probability, not conditioning on the class.

**Correct formula:**

$$P(Y,X_1,X_2,X_3)=P(Y)P(X_1\mid Y)P(X_2\mid Y)P(X_3\mid Y).$$

**Reason:** Conditional independence gives

$$P(X_1,X_2,X_3\mid Y)=\prod_{j=1}^{3}P(X_j\mid Y).$$

Multiplying by P(Y) gives the joint distribution. Using joint factors P(X2,Y) and P(X3,Y) instead would introduce extra class-prior factors and generally would not give the intended joint distribution.

The TAN formula is correctly written:

$$P(Y,X_1,X_2,X_3)=P(Y)P(X_1\mid Y)P(X_2\mid Y,X_1)P(X_3\mid Y,X_2).$$

The independence assumption, plausible HAR feature-dependence example, and parameter/structure-learning trade-off are correct. The written explanation indicates a transcription error in the NB formula rather than a demonstrated conceptual misunderstanding.

## Review outcome and provenance

All four topics are addressed, with one notation correction in Q4. The scan remains unchanged. Verification consists of recomputing the supplied GA operations, parent-based factorization/parameter count, d-separation paths, and the conditional-independence algebra above against the worked drill in this conversation.

This review does not assign an instructor score or claim independent reproduction. The scan has its own commit above; this corrections file belongs to the later revision commit. W07 progress fields record the actual evidence hashes.
