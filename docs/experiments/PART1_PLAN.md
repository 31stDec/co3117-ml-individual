# Part I practice plan - recorded before running on 2026-10-14

These are new Codex-assisted experiments, not retroactive first-attempt evidence. The old HW2/HW3 numbers describe separate historical library runs and are not replaced by these experiments. The two final PDF documents are deferred at the user's request.

## Fixed evaluation

Use the existing train-subject holdout: 15 fit subjects, six validation subjects, seed 42, labels 1--6, Macro-F1 primary and accuracy/confusion matrices secondary. Never load the final test files. Fit scaling, ANOVA ranking and any search fitness split on fit data only. Report validation results as validation, not final test performance.

## Tree

Study the saved ML-From-Scratch recursive classification-tree reference. Adapt it to continuous HAR with midpoint thresholds, vectorized information gain, exact depth stopping, and a minimum child-size rule. Compare unlimited depth versus depth 6 (minimum leaf size 1), then unlimited depth with minimum leaf size 20. Also benchmark sklearn entropy trees with ccp_alpha 0 and 0.002.

Prediction: unrestricted trees will fit training data best. Depth/leaf restrictions and pruning will shrink trees and reduce training performance; validation may improve or worsen depending on the bias/variance trade-off. Record depth, leaves, scores, time and important split features.

## MLP

Use the existing NumPy tanh/softmax forward and backward functions with mini-batch SGD. Compare one hidden layer of 32 versus 64 units, using seeds 42, 0, 1; 60 epochs, batch size 128, learning rate 0.05, no early stopping. The shuffle sequence is fixed within each seed pair. Scale using outer fit rows only. Preserve losses across epochs, train/validation scores, confusion matrices, parameters and runtime.

Prediction: 64 units will generally fit training data better and take longer. More capacity does not guarantee better validation performance. This SGD experiment is different from the historical HW3 sklearn Adam experiment.

## GA

Study the saved ML-From-Scratch string-search reference, adapting its population/crossover/mutation loop to binary feature masks. Restrict to 12 features ranked by ANOVA on an inner fit population inside the 15 outer fit subjects. Fitness = inner validation Macro-F1 - 0.02 * selected_count / 12, using GaussianNB. Tournament selection supports nonpositive fitness. Repair empty masks, cache repeated fitness computations, seed all randomness, and track diversity and an all-time best mask.

Compare no elitism with one unchanged elite per generation, seeds 42, 0, 1, population 16, 12 evaluated generations, bit mutation probability 1/12. Identical initial populations within seed pairs. Refit the chosen mask on outer fit rows and evaluate once on the same outer validation. Compare all 12 candidates as a baseline.

Prediction: elitism will prevent the generation-best fitness from falling, but may reduce diversity and need not improve outer validation. Three seeds are a small robustness check, not a confidence interval or proof of optimality. GA does not search all 561 features and no final test score will be added.
