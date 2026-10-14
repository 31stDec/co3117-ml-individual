# MODEL_LOG

One block per model family (assignment Section 9.1). Copy the block below.

## Template

### <Model name>, Chapter <n>, depth <A/B/C>

- Objective / factorization / decision rule:
- Assumptions, inductive bias, expected failure modes:
- Representation and preprocessing:
- Own pre-reference commit; reference code consulted (repository, file, function, commit or version):
- Hyperparameters and selection procedure:
- Primary / secondary metrics, runtime, model size:
- Diagnostic plot or table; focused experiment:
- Error and limitation analysis; use-case fit:
- Exam-ready paragraph (no code):

---

## Logged Models

### Majority & Random Baselines, Chapter 1, depth Baseline
- Objective / factorization / decision rule: Majority baseline always predicts the most frequent training class ($y = \text{argmax}_c N_c$, which is class 6 LAYING). Random baseline samples classes from empirical training prior distribution $P(y=c)$.
- Assumptions, inductive bias, expected failure modes: Naive lower-bound benchmarks assuming feature data contains zero predictive signal. Fails dramatically on balanced multiclass problems.
- Representation and preprocessing: Raw integer class labels $\{1, \dots, 6\}$; feature data ignored.
- Own pre-reference commit; reference code consulted: `ec63e23`; custom implementation in `experiments/part1_pre_midterm/w05_baseline.py`.
- Hyperparameters and selection procedure: Random seed 42.
- Primary / secondary metrics, runtime, model size: Majority: Macro-F1 = 0.0514, Accuracy = 0.1822; Random: Macro-F1 = 0.1618, Accuracy = 0.1639; Both baselines require negligible fitting time.
- Diagnostic plot or table; focused experiment: Reported in `results/metrics.csv` and `docs/weekly/w05-perceptron-ann.md`.
- Error and limitation analysis; use-case fit: Serves as sanity floor; cannot distinguish dynamic activities (Walking, Walking Upstairs) from static postures (Sitting, Standing).
- Exam-ready paragraph (no code): Majority class prediction establishes the absolute trivial performance floor for classification. In an imbalanced or multi-class setting, accuracy can be misleading, but Macro-F1 severely penalizes trivial majority guessing because recall on all minority classes drops to zero. A valid machine learning model must convincingly surpass both majority and stratified empirical baselines.

### MultiClassPerceptron (One-vs-Rest), Chapter 4, depth A
- Objective / factorization / decision rule: Binary Perceptron uses step threshold $\hat{y} = \text{sign}(w \cdot x + b)$, updated strictly on error: $w \leftarrow w + \eta y x$, $b \leftarrow b + \eta y$. Multi-class extension trains 6 One-vs-Rest binary perceptrons and predicts $\hat{c} = \arg\max_{k \in \{1,\dots,6\}} (w_k \cdot x + b_k)$.
- Assumptions, inductive bias, expected failure modes: Assumes linear separability per binary OvR split. Inductive bias is a linear hyperplane separating each class from the rest. Expected failure mode: cycling without convergence on non-linearly separable or overlapping distributions.
- Representation and preprocessing: 561-dimensional smartphone inertial feature vector; Standard z-score scaling fit strictly on train split ($X' = (X - \mu)/\sigma$).
- Own pre-reference commit; reference code consulted: `48b97e7`; Tom Mitchell (1997) *Machine Learning*, Chapter 4, Section 4.5.
- Hyperparameters and selection procedure: Learning rate $\eta = 0.01$, max epochs = 100, random seed = 42; historical fixed configuration. The original sweep evaluated the test set and was not a valid validation procedure. On 14 Oct, a subject-held-out validation sweep gave identical predictions at all five positive learning rates; no improvement from learning-rate tuning was demonstrated.
- Primary / secondary metrics, runtime, model size: 14 Oct fixed-configuration rerun: test Macro-F1 = 0.9545, test accuracy = 0.9539 (previously exposed test); validation Macro-F1 = 0.9496; Rerun training time = 7.09s; Model size = 6 weight vectors $\times$ 561 floats + 6 biases (~27 KB).
- Diagnostic plot or table; focused experiment: Validation sweep and confusion matrices in `results/perceptron_validation_audit.json`; AND vs XOR synthetic boundary checks in `src/from_scratch/perceptron.py`.
- Error and limitation analysis; use-case fit: The fixed linear model performs well on these engineered features; this result alone does not establish why separation works. Main residual confusion occurs between Sitting and Standing due to subtle acceleration differences.
- Exam-ready paragraph (no code): The classic Perceptron is a linear threshold classifier that updates weights only when an example is misclassified. According to the Perceptron Convergence Theorem, it is guaranteed to locate a separating hyperplane in finite steps if the data is linearly separable, but cycles indefinitely when classes overlap. To solve multi-class problems such as human activity recognition, the One-vs-Rest paradigm trains independent binary detectors and resolves competitive classification by taking the maximum continuous signed distance to each decision boundary.

### GaussianNB, Chapter 6, depth A
- Objective / factorization / decision rule: MAP classification under conditional independence: $\hat{y} = \arg\max_c \log P(c) + \sum_j \log P(x_j|c)$. Each feature likelihood is a univariate Gaussian $\mathcal{N}(\mu_{cj}, \sigma^2_{cj})$.
- Assumptions, inductive bias, expected failure modes: Assumes features are conditionally independent given the class. Inductive bias is axis-aligned Gaussian clusters. Expected failure: correlated features (e.g. polynomial interactions) degrade probability calibration, though ranking may remain correct.
- Representation and preprocessing: Continuous feature vectors; no mandatory scaling (generative model). Population per-class variance (ddof=0) with smoothing based on the largest feature variance over all fit rows, including between-class variation.
- Own pre-reference commit; reference code consulted: `b60f5c5`; sklearn source (`sklearn.naive_bayes.GaussianNB`); Tom Mitchell (1997) Chapter 6.
- Hyperparameters and selection procedure: `var_smoothing=1e-9` (default); no tuning needed for contract tests.
- Primary / secondary metrics, runtime, model size: HAR validation Macro-F1 0.722378, accuracy 0.736311; sklearn prediction agreement 100%; log-posterior difference about 5.59e-9. Fit time 0.0228 s in the metric audit; means and variances require 6 x 561 x 2 floats plus priors.
- Diagnostic plot or table; focused experiment: Comparison table in `docs/weekly/w06-naive-bayes.md` Section D.
- Error and limitation analysis; use-case fit: GNB is fast ($O(ndk)$) and works well on high-dimensional data with approximately independent features. Poor calibration on correlated features but decision boundaries remain robust for classification.
- Exam-ready paragraph (no code): Gaussian Naive Bayes classifies by computing MAP posteriors under a conditional-independence assumption, modelling each feature per class as a univariate Gaussian with population variance. Variance smoothing adds a data-scaled epsilon to prevent singularity on constant features. Despite the naive independence assumption, GNB is competitive because correct posterior ranking suffices for classification accuracy. Training is a single pass computing per-class sufficient statistics, making it one of the fastest supervised learners.


### Continuous Decision Tree, Chapter 2, depth B

- Objective / decision rule: Recursively maximize information gain, H(parent) - weighted child entropy; predict the majority training class at each leaf.
- Assumptions / inductive bias / failure modes: Axis-aligned partitions, piecewise-constant predictions; deep trees can memorize fit subjects and vary with split ties.
- Representation / preprocessing: All 561 continuous HAR features, no scaling; the same 15/6 outer subject split as other validation models.
- Provenance / reference: Existing impurity core in `src/from_scratch/tree_split.py`; new Codex-assisted adapter in `src/reference_adapters/decision_tree.py`. Saved MIT ML-From-Scratch source, exact hashes and line mapping in `docs/reference/DISSECTION.md`. No independent pre-reference authorship is claimed for the new adapter.
- Hyperparameters / selection: Prespecified unlimited versus max_depth=6, then min_samples_leaf=20; no post-hoc final-test selection. sklearn entropy-tree ccp_alpha=0 versus 0.002 is a separate pruning benchmark.
- Metrics / runtime / size: Adapted full: train F1 1.0000, validation F1 0.862417, depth 14, 100 leaves. Depth six: train F1 0.9529, validation F1 0.871938, 38 leaves. Leaf size 20: validation F1 0.854601, 54 leaves. sklearn pruning reduces 100 to 65 leaves but validation F1 falls 0.881521 to 0.870621. Exact times and validation accuracy are in `results/tree_validation_audit.json`.
- Diagnostic / focused experiment: `results/figures/tree_stopping_pruning.png`; controlled depth, child-size and pruning comparisons. Top adapted features are tBodyAccJerk-std()-X, tGravityAcc-min()-X and fBodyAccMag-std(). Feature indices disambiguate repeated HAR names.
- Error / limitation / use-case fit: Per-run confusion matrices and the largest directed confusion are saved. Depth restriction helps this holdout slightly, while other regularizers do not. sklearn and the adaptation need not resolve tied features identically. One holdout is insufficient to establish an optimal tree setting.
- Exam-ready paragraph: A decision tree partitions feature space by greedily reducing label impurity. Pure fit leaves can overfit. Early stopping limits depth or leaf size; post-pruning trades fit loss against complexity. Each intervention should be evaluated on validation subjects rather than assumed to improve generalization.

### NumPy MLP / Backpropagation, Chapter 3, depth A core

- Objective / decision rule: Minimize mean multiclass cross-entropy with tanh hidden units and stable softmax output; predict the largest class probability.
- Assumptions / inductive bias / failure modes: Nonlinear parametric classifier; optimization depends on initialization and learning dynamics. More hidden units can fit noise and do not guarantee better validation.
- Representation / preprocessing: All 561 features, z-score fitted on outer fit rows only; six output classes. One hidden layer, widths 32 and 64.
- Provenance / reference: Existing explicit forward/backward core in `src/from_scratch/mlp.py`, with numerical-gradient contract tests. New Codex-assisted mini-batch SGD wrapper in `src/nn_training.py`. These new experiments are not the historical HW3 sklearn/Adam runs, and independent ownership of the added wrapper is not claimed.
- Hyperparameters / selection: Prespecified widths 32/64, lr=0.05, batch=128, 60 epochs, no early stopping; paired shuffle seeds 42, 0, 1. Width is the only varied capacity setting.
- Metrics / runtime / size: Width 32: validation F1 mean 0.949486, sample SD 0.008478, 18,182 parameters, mean fit time 1.24 s. Width 64: mean 0.950444, SD 0.004050, 36,358 parameters, mean time 2.10 s. Individual train/validation accuracy and F1 are in `results/mlp_validation_audit.json`.
- Diagnostic / focused experiment: `results/figures/mlp_capacity.png` and 60-epoch train/validation loss histories. Core gradients match numerical differentiation in the provided contract tests; a new synthetic training test verifies loss reduction and classification.
- Error / limitation / use-case fit: All confusion matrices and largest directed error counts are saved. Width 64 fits training data better in all three seed pairs, but two pairs favor width 32 on validation. The 0.001 mean advantage is smaller than observed seed variation; no robust capacity benefit is established. BLAS is limited to one thread for paired runtimes.
- Exam-ready paragraph: Backpropagation applies the chain rule from the output loss to each layer's weights. Nonlinear hidden activations allow boundaries a single linear classifier cannot represent. Mini-batch SGD uses batch gradients to update parameters. Capacity affects fit and computation, while generalization must be checked separately.

### Genetic Algorithm Feature Selection, Chapter 5, depth B

- Objective / decision rule: Search binary feature masks; fitness = inner validation Macro-F1 - 0.02 * selected_count/12. Predict activities using GaussianNB with the chosen feature subset.
- Assumptions / inductive bias / failure modes: Random bounded search can converge prematurely, and a good inner-validation mask may transfer poorly to other subjects. The fitness trades performance for compression.
- Representation / preprocessing: Twelve candidate features ranked by ANOVA on inner-fit rows only. Inner split uses 11/4 subjects within the 15 outer-fit subjects; final mask refit uses all outer-fit rows and is evaluated on the same six outer validation subjects. No final-test files are read.
- Provenance / reference: Codex-assisted adaptation of MIT ML-From-Scratch target-string GA to feature masks; attribution, source hash, tournament-selection change, empty-mask repair, caching and elite-retention mapping in `docs/reference/DISSECTION.md`.
- Hyperparameters / selection: Population 16, 12 evaluated generations, mutation=1/12, tournament size three, seeds 42, 0, 1. Compare elite_count=0 versus 1 with paired identical initial populations. Both return the best mask seen across all generations.
- Metrics / runtime / size: Five runs choose seven features: best inner fitness 0.674613 and outer F1 0.731027. Seed 1 with one elite chooses eight: inner fitness 0.672145 and outer F1 0.740619. All twelve candidates give outer F1 0.740683. JSON records search/refit time separately, unique fitness evaluations and selected feature names; no performance gain over all twelve is demonstrated.
- Diagnostic / focused experiment: `results/figures/ga_elitism_diversity.png`; full current-best/all-time-best fitness and normalized Hamming diversity histories in `results/ga_validation_audit.json`.
- Error / limitation / use-case fit: Elitism prevents current-generation best fitness from decreasing but is not uniformly better within the budget. Seed 1 illustrates disagreement between inner fitness ranking and outer performance. Search explores only a subset of the 12-bit space, not all 561 HAR features; three seeds do not prove optimality. Per-run confusion matrices document activity errors.
- Exam-ready paragraph: A GA represents candidate solutions as chromosomes, chooses parents, recombines their alleles and applies mutation. Encoding, fitness and operators determine its search behavior. Elitism protects a good candidate, while diversity supports exploration. Repeated seeds and held-out subjects are needed to distinguish a fortunate search from a transferable result.

### Bayesian Networks / TAN, Chapter 6, depth C structural suitability study

- Objective / factorization: P(A,B,C,D,E)=P(A)P(B)P(C|A,B)P(D|C)P(E|B). For the three-feature TAN, P(Y,X)=P(Y)P(X1|Y)P(X2|Y,X1)P(X3|Y,X2).
- Assumptions / inductive bias / failure modes: DAG-based conditional factorization; TAN permits one additional feature parent per node in a tree. It cannot represent arbitrary higher-order dependencies.
- Representation / preprocessing: Illustrative five-binary-variable CPTs verify the written derivation; HAR evidence uses continuous magnitude mean/energy on outer fit rows. The illustration is not a replacement prediction dataset.
- Provenance / reference: AI-assisted worked drill, preserved scan and notation correction. Direct numerical enumeration of all 32 states in `w07_bn.py`; no fitted TAN classifier is submitted or claimed.
- Parameters / procedure: Ten independent parameters for the specified binary graph. Fixed CPTs check the unobserved collider, conditioning on C, conditioning on descendant D, and conditioning on the chain middle node.
- Metrics / evidence: All four numerical conditional-independence checks pass; joint sum is one. HAR within-class correlations for tBodyAccMag-mean() versus tBodyAccMag-energy() range from 0.8676 to 0.9884. No TAN accuracy or runtime benchmark is claimed.
- Diagnostic / focused study: `results/bn_tan_checks.json`, the drill and `results/PART1_PRACTICE_RESULTS.md`. Correlation is measured within each class on fit rows, not inferred from aggregate correlations alone.
- Limitation / use-case fit: Strong within-class correlation challenges NB independence, but does not prove a fitted TAN would improve. Continuous TAN needs conditional density estimation; discretization would require train-only bins and sacrifices information. This is a structural suitability study rather than empirical TAN-versus-NB classification.
- Exam-ready paragraph: A Bayesian network factors the joint using each node's parents. An unobserved collider blocks a path, while evidence at the collider or its descendant can activate it. Naive Bayes requires feature independence given the class. TAN relaxes that restriction along a feature tree and pays for extra conditional parameters.

## Dated correction -- 2026-10-14

The earlier metric included unused label 0. Current tables use HAR labels 1--6. Original recorded values and reproducibility limitations are preserved in `results/PART1_METRIC_AUDIT.md` and Git history. GaussianNB now uses `var_smoothing * X.var(axis=0).max()` over all fit rows. Its subject-held-out validation Macro-F1 is 0.722378, accuracy 0.736311; predictions agree 100% with the installed sklearn reference. The largest directed error is STANDING to SITTING (264 cases). Mean maximum posterior is 0.9971 while accuracy is 0.7363; log loss is 7.1509, showing that normalized posteriors are not necessarily calibrated. No final-test benchmark for GNB was added.
