# W09 - Hidden Markov Models

Date: 2026-10-27. Status: draft for review.

**Question I am trying to answer:** Does modeling temporal dependence improve HAR predictions compared with treating each window independently?

Working hypothesis, recorded retrospectively: persistence should reduce isolated errors within an activity. It may also propagate a wrong prediction across a segment. This is not a recorded pre-experiment hypothesis.

## A. Concept capsule

An HMM contains initial state probabilities pi, a transition matrix A, and an emission matrix B. The next state depends only on the current state, and each observation depends on its corresponding hidden state. Its joint distribution is:

$$P(q_{1:T},O_{1:T})=\pi_{q_1}b_{q_1}(O_1)\prod_{t=2}^{T}a_{q_{t-1},q_t}b_{q_t}(O_t).$$

Here, states represent six activities and observations are discrete symbols derived from sensor features.

## B. One derivation or worked example

The [handwritten drill](../../exercises/w09-first-attempt.pdf) uses two urns and observations (R,R). Initialization gives alpha_1=(0.48,0.04). Applying the Forward recurrence gives:

$$\alpha_2(1)=[0.48(0.7)+0.04(0.2)](0.8)=0.2752,$$
$$\alpha_2(2)=[0.48(0.3)+0.04(0.8)](0.1)=0.0176.$$

Therefore P(R,R)=0.2928. The arithmetic is correct; [corrections](../../exercises/w09-corrections.md) add missing equations and clarify notation.

## C. Code-to-theory trace

In [hmm.py](../../src/from_scratch/hmm.py), lines 51-56 implement Forward prediction, emission weighting, and normalization. Line 57 sums log scaling factors to recover log-likelihood. Lines 79-81 implement the Viterbi maximum and predecessor selection; lines 88-89 backtrack.

The [pinned numpy-ml comparison](../reference/W09_HMM_DISSECTION.md) maps these equations to upstream `_forward` lines 343-359 and `decode` lines 265-289. Upstream stores log joint probabilities; our Forward returns normalized filtering probabilities. Fifteen numerical comparisons pass, allowing equally probable Viterbi paths. Ten automated tests pass, including exhaustive-enumeration, boundary, and leakage checks.

## D. One controlled experiment

The [experiment](../../experiments/part2_post_midterm/w09_hmm.py) fits 5,270 windows from 15 subjects and validates on 2,082 windows from six other subjects. No official test data is loaded. Standardization and 32-centroid quantization are fitted only on training subjects. Add-one-smoothed labeled counts estimate pi, A, and B; Baum-Welch is not used.

Windows join only when subject identity and overlapping halves match across all nine inertial channels. Labels do not determine boundaries. Independent inference, filtering, and Viterbi share pi and B; the comparison changes how transitions are used. Five quantizer seeds share one fixed subject split.

| Method | Mean Macro-F1 | Mean accuracy |
|---|---:|---:|
| Independent | 0.725746 | 0.727570 |
| Filtering | 0.798413 | 0.801921 |
| Viterbi | 0.828615 | 0.832181 |

[Results](../../results/w09_hmm/REPORT.md) include all seeds, confusion matrices, and predictions. Seed variation is not a confidence interval across subject populations.

## E. Failure / misconception

With seed 42, Viterbi corrects 283 independent errors but introduces 103. Standing-to-sitting confusion affects 90 validation windows. Temporal persistence can reinforce an incorrect state. Verified segments contain no activity changes, so this evaluates within-activity denoising, not transition detection. Overlapping windows also challenge conditional emission independence.

## F. Written-exam capsule

Forward sums over possible hidden paths to compute observation likelihood. Viterbi maximizes joint path probability and stores predecessor pointers. Backtracking recovers one consistent optimal path. Filtering estimates the current state from observations so far. Viterbi uses the complete sequence and is therefore offline. Scaling or log-space prevents numerical underflow.

## G. Reflection

The key distinction is between unscaled Forward values and normalized filtering probabilities. Better segment accuracy does not establish better detection of activity changes. Independent reproduction of the derivations remains to be checked; transition behavior remains untested on this reconstructed view.

## H. Inquiry trail

For the handwritten review, Codex was asked to check the answers against the HMM slides. The scan is archived at `969096b`; this identifies the reviewed artifact, not pre-AI provenance. Verification used lecture pages 8, 13, and 20-25 and direct arithmetic. The feedback confirmed the numbers and added missing formulas. Closed-book reproduction has not yet been recorded. This paragraph covers the handwritten-review episode; the wider W09 assistance record is still pending completion.
