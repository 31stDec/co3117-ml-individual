# W09 Written Drill - Review and Corrections

Reviewed on: 2026-10-27.
Original: [Handwritten HMM drill](w09-first-attempt.pdf), three pages, scan commit `969096b`.
Reference: HCMUT lecture slides, *Hidden Markov Models*, 11 August 2026, pages 8, 13, and 20-25 (`HMM.pdf`).

## Q1. HMM basics

The definitions of hidden states (urns), observations (colors), initial probabilities, transitions, and emissions are correct. The row-sum conditions for A and B are also correct. The initial probabilities must satisfy:

$$\sum_{i=1}^{N}\pi_i=1.$$

**Addition for the W09 syllabus:** the joint factorization was not included in the written answer:

$$P(O_{1:T},q_{1:T}\mid\lambda)
=\pi_{q_1}b_{q_1}(O_1)
\prod_{t=2}^{T}a_{q_{t-1},q_t}b_{q_t}(O_t).$$

This uses two assumptions: the next state depends only on the current state, and each observation depends only on its corresponding hidden state.

## Q2. Forward calculation

The numerical calculations on page 2 are correct:

$$\alpha_1=(0.48,0.04),$$

$$\alpha_2(1)=[0.48(0.7)+0.04(0.2)](0.8)=0.2752,$$

$$\alpha_2(2)=[0.48(0.3)+0.04(0.8)](0.1)=0.0176,$$

$$P(R,R\mid\lambda)=0.2752+0.0176=0.2928.$$

No arithmetic correction is needed. In the general recurrence, keep the destination state fixed and sum over the previous state:

$$\alpha_t(i)=\left[\sum_{k=1}^{N}\alpha_{t-1}(k)a_{ki}\right]b_i(O_t).$$

**Code-to-notation clarification:** these handwritten alpha values are joint probabilities, not normalized state probabilities. The repository's `forward` interface returns filtered probabilities instead:

$$\widehat\alpha_t(i)=\frac{\alpha_t(i)}{\sum_j\alpha_t(j)}.$$

For example, the final filtered probabilities are approximately `(0.93989, 0.06011)`. Scaling during the recursion avoids underflow; the log-likelihood is recovered by adding the logarithms of the scaling factors.

## Q3. Forward versus Viterbi

The written distinction is correct: Forward sums over all state paths to compute the observation likelihood; Viterbi maximizes over paths to find the most probable complete state sequence.

**Missing detail:** the Viterbi recurrence requested in the study draft is:

$$\delta_1(i)=\pi_i b_i(O_1),$$

$$\delta_t(i)=\max_k[\delta_{t-1}(k)a_{ki}]b_i(O_t).$$

Here, delta stores the best joint probability of a path and the observations so far, subject to ending in state i. It is not the filtered probability of state i.

## Q4. Viterbi backtracking

The explanation on page 3 is correct. The predecessor table records which previous state produced the best path into each current state:

$$\phi_t(i)=\arg\max_k[\delta_{t-1}(k)a_{ki}].$$

Choose the final state, then follow the stored pointers:

$$q_T^*=\arg\max_i\delta_T(i),\qquad
q_t^*=\phi_{t+1}(q_{t+1}^*).$$

**Precision:** choosing the largest delta independently at each time can splice together states from different best prefixes. The resulting sequence may be suboptimal, or impossible if a selected transition has zero probability. Backtracking preserves a consistent optimal path.

## Review outcome

The main explanations and Forward arithmetic are correct. The additions are the joint factorization, explicit Viterbi equations, and the distinction between unscaled Forward values and normalized filtering probabilities. The original handwritten scan is preserved.
