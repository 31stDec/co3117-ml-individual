# W13 HAR LDA Experiment Report

Best model: `lda_k5_svc` with validation Macro-F1 = 0.966564 (Accuracy = 0.969260).

## Key Comparisons
- Baseline Linear SVC (561 features): Macro-F1 = 0.947470
- PCA (k=5 features) + SVC: Macro-F1 = 0.815499
- LDA (k=5 features) + SVC: Macro-F1 = 0.966564
- Direct LDA (SVD): Macro-F1 = 0.964329

LDA compresses 561 dimensions down to just 5 discriminative axes while improving classification performance over the full-dimensional baseline.