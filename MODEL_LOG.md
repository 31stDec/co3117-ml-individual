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
- Primary / secondary metrics, runtime, model size: Majority: Macro-F1 = 0.0440, Accuracy = 0.1822; Random: Macro-F1 = 0.1387, Accuracy = 0.1639; Runtime < 0.01s.
- Diagnostic plot or table; focused experiment: Reported in `results/metrics.csv` and `docs/weekly/w05-perceptron-ann.md`.
- Error and limitation analysis; use-case fit: Serves as sanity floor; cannot distinguish dynamic activities (Walking, Walking Upstairs) from static postures (Sitting, Standing).
- Exam-ready paragraph (no code): Majority class prediction establishes the absolute trivial performance floor for classification. In an imbalanced or multi-class setting, accuracy can be misleading, but Macro-F1 severely penalizes trivial majority guessing because recall on all minority classes drops to zero. A valid machine learning model must convincingly surpass both majority and stratified empirical baselines.

### MultiClassPerceptron (One-vs-Rest), Chapter 4, depth A
- Objective / factorization / decision rule: Binary Perceptron uses step threshold $\hat{y} = \text{sign}(w \cdot x + b)$, updated strictly on error: $w \leftarrow w + \eta y x$, $b \leftarrow b + \eta y$. Multi-class extension trains 6 One-vs-Rest binary perceptrons and predicts $\hat{c} = \arg\max_{k \in \{1,\dots,6\}} (w_k \cdot x + b_k)$.
- Assumptions, inductive bias, expected failure modes: Assumes linear separability per binary OvR split. Inductive bias is a linear hyperplane separating each class from the rest. Expected failure mode: cycling without convergence on non-linearly separable or overlapping distributions.
- Representation and preprocessing: 561-dimensional smartphone inertial feature vector; Standard z-score scaling fit strictly on train split ($X' = (X - \mu)/\sigma$).
- Own pre-reference commit; reference code consulted: `48b97e7`; Tom Mitchell (1997) *Machine Learning*, Chapter 4, Section 4.5.
- Hyperparameters and selection procedure: Learning rate $\eta = 0.01$, max epochs = 100, random seed = 42; selected via validation curve sweep across $\eta \in [0.001, 0.1]$.
- Primary / secondary metrics, runtime, model size: Test Macro-F1 = 0.8143, Test Accuracy = 0.9494; Training time = 12.50s; Model size = 6 weight vectors $\times$ 561 floats + 6 biases (~27 KB).
- Diagnostic plot or table; focused experiment: Learning rate comparison table in `docs/weekly/w05-perceptron-ann.md`; AND vs XOR synthetic boundary checks in `src/from_scratch/perceptron.py`.
- Error and limitation analysis; use-case fit: Perceptron achieves high accuracy due to high dimensionality ($d=561$), confirming Cover's Theorem. Main residual confusion occurs between Sitting and Standing due to subtle acceleration differences.
- Exam-ready paragraph (no code): The classic Perceptron is a linear threshold classifier that updates weights only when an example is misclassified. According to the Perceptron Convergence Theorem, it is guaranteed to locate a separating hyperplane in finite steps if the data is linearly separable, but cycles indefinitely when classes overlap. To solve multi-class problems such as human activity recognition, the One-vs-Rest paradigm trains independent binary detectors and resolves competitive classification by taking the maximum continuous signed distance to each decision boundary.
