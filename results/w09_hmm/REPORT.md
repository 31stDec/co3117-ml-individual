# W09 HAR validation experiment

Run: `python -m experiments.part2_post_midterm.w09_hmm`

Fixed subject split; official test files are not loaded. Quantizer seeds vary; the split remains fixed.

| Seed | Model | Macro-F1 | Accuracy |
|---|---|---:|---:|
| 42 | independent | 0.726809 | 0.731988 |
| 42 | filtering | 0.806578 | 0.808357 |
| 42 | viterbi | 0.818376 | 0.818444 |
| 0 | independent | 0.754490 | 0.756004 |
| 0 | filtering | 0.810167 | 0.813641 |
| 0 | viterbi | 0.847151 | 0.851585 |
| 1 | independent | 0.680830 | 0.683477 |
| 1 | filtering | 0.765825 | 0.772334 |
| 1 | viterbi | 0.818120 | 0.825648 |
| 2 | independent | 0.735696 | 0.735351 |
| 2 | filtering | 0.806651 | 0.809798 |
| 2 | viterbi | 0.828660 | 0.832373 |
| 3 | independent | 0.730904 | 0.731028 |
| 3 | filtering | 0.802845 | 0.805476 |
| 3 | viterbi | 0.830770 | 0.832853 |

## Across quantizer seeds

Values are mean +/- population SD; these are not independent subject splits or confidence intervals.

- independent: Macro-F1 0.725746 +/- 0.024374; accuracy 0.727570 +/- 0.023852.
- filtering: Macro-F1 0.798413 +/- 0.016458; accuracy 0.801921 +/- 0.015026.
- viterbi: Macro-F1 0.828615 +/- 0.010613; accuracy 0.832181 +/- 0.011031.

## Sequence audit

```json
{
  "fit": {
    "sequences": 424,
    "min_length": 2,
    "median_length": 12.0,
    "max_length": 29,
    "singletons": 0,
    "verified_links": 4846,
    "label_changes_on_links": 0
  },
  "validation": {
    "sequences": 176,
    "min_length": 1,
    "median_length": 12.0,
    "max_length": 25,
    "singletons": 1,
    "verified_links": 1906,
    "label_changes_on_links": 0
  }
}
```

## Failure examples (seed 42)

First five remaining Viterbi errors in original row order, not selected for appearance.

| Source row | Subject | True activity | Independent | Viterbi |
|---|---|---|---|---|
| 1 | 1 | STANDING | STANDING | SITTING |
| 2 | 1 | STANDING | STANDING | SITTING |
| 3 | 1 | STANDING | STANDING | SITTING |
| 4 | 1 | STANDING | STANDING | SITTING |
| 5 | 1 | STANDING | STANDING | SITTING |

## Interpretation limits

Sequence boundaries come only from matching sensor overlap and subject identity. Labels are inspected afterward for this audit, not used to choose boundaries.

Verified chains contain no observed activity changes. This evaluates within-activity denoising, not transition detection. The smoothed off-diagonal transition probabilities come from the add-one prior, not observed transitions. Overlapping windows also violate strict conditional emission independence, so likelihoods are not calibrated evidence for independent samples.

Viterbi uses the full segment and is offline; filtering uses only the prefix. Quantization loses feature information. This comparison isolates sequence inference within one categorical-emission model, not superiority over all existing HAR classifiers.

See audit.json for all confusion matrices, parameters, input hashes, and the pinned numpy-ml numerical comparison; predictions.csv contains every validation prediction.
