# REFERENCES

Cite every source you actually used, with a locator (chapter and section, lecture number and minute,
repository path and commit). One line per source.

## Course Resources & Repositories

### W12 PCA -- 2026-11-17

- HCMUT PCA lecture, 11 August 2026, local `C:/Users/USER/Downloads/ML/PCANew.pdf`: slides 4-5 (dimensionality), 19-26 (centering and variance objective), 31-33 (diagonalization), 37-41 (projection/reconstruction), 42-46 (SVD).
- Erik Linder-Noren, ML-From-Scratch, commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`, `mlfromscratch/unsupervised_learning/principal_component_analysis.py` and `mlfromscratch/utils/data_operation.py::calculate_covariance_matrix`. https://github.com/eriklindernoren/ML-From-Scratch/tree/a2806c6732eee8d27762edd6d864e0c179d8e9e8 . MIT source snapshots, license and hashes in `docs/reference/pca/`. Exact AST definitions executed with dependencies injected; comparison details in `results/w12_pca/audit.json`.
- Installed sklearn 1.9.1 `decomposition.PCA._fit_full`, inspected for centering, full SVD, sample variance and component signs; local source hash and numerical comparisons in `results/w12_pca/audit.json`. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html . NumPy SVD supplies the decomposition for the course implementation.
- Course PCA interface and `tests/contract/test_pca.py`: required row-oriented components and ddof=1 variances. Additional edge-case tests are in `tests/test_pca_edge_cases.py`.
- UCI HAR v1.0, unchanged subject-held-out split; input hashes and baseline predictions checked against W10. The W12 experiment reads only the original training partition.

- ML-From-Scratch: https://github.com/eriklindernoren/ML-From-Scratch
- numpy-ml: https://github.com/ddbourgin/numpy-ml
- pyprobml: https://github.com/probml/pyprobml
- NPTEL Introduction to Machine Learning (IIT Madras): https://nptel.ac.in/courses/106106139
- UCI Human Activity Recognition Using Smartphones: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- scikit-learn documentation: https://scikit-learn.org/stable/

### W11 Kernel SVM -- 2026-11-09

- HCMUT SVM lecture, 11 August 2026, slide labels 23-27 (dual), 28-33 (kernel trick), 34-35 (soft margin); local `C:/Users/USER/Downloads/ML/SVM.pdf`. RBF is an assignment extension to the lecture's polynomial example.
- ML-From-Scratch, commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`, `mlfromscratch/utils/kernels.py` and `mlfromscratch/supervised_learning/support_vector_machine.py`. Existing MIT snapshots and hashes: `docs/reference/svm/provenance.json`. W11 executes the kernel helpers; it does not execute CVXOPT.
- Installed sklearn SVC and `metrics.pairwise.rbf_kernel` source/docstrings, inspected 2026-11-09; https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html and https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html . Version and SVC source hash: `results/w11_kernel_svm/audit.json`. GaussianNB and DummyClassifier provide fixed-setting comparison baselines.
- UCI HAR v1.0 and the frozen subject split; W10 input hashes, validation row IDs, and predictions are checked before cross-week comparison. No official test files are read in W11.

## Course Textbooks & Materials
- [1] CO3117 Machine Learning course syllabus, HK261, Faculty of Computer Science and Engineering, HCMUT, VNU-HCM.
- [2] Sample CO3117 final examination, question sheet code 504, Semester 2 2025–2026, 26 May 2026.
- [3] Andreas C. Müller and Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2017.
- [4] François Chollet, *Deep Learning with Python*, 2nd ed., Manning, 2021.
- [5] Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.
- [6] Cao Hoàng Trụ, *Trí tuệ Nhân tạo = Thông minh + Giải thuật*, NXB ĐHQG-HCM, 2008.
- [7] Tom M. Mitchell, *Machine Learning*, McGraw-Hill, 1997.
- [8] Stephen Marsland, *Machine Learning: An Algorithmic Perspective*, Chapman & Hall/CRC, 2009.
- [9] Instructor-provided prior-cohort exemplar: CO3101 PIP, *Detect AI Generated Text* (2024).

## Weekly Specific Citations

### W10 SVM -- 2026-11-03

- HCMUT, Faculty of Computer Science and Engineering, *Support Vector Machines*, 11 August 2026. Slide numbers 5-13: hard margin and geometry; 34-35: soft margin and C. Local file `C:/Users/USER/Downloads/ML/SVM.pdf`; slide numbers differ from PDF page numbers.
- Erik Linder-Noren, ML-From-Scratch, commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`: `mlfromscratch/supervised_learning/support_vector_machine.py`, `SupportVectorMachine.fit` and `predict`; `mlfromscratch/utils/kernels.py`, `linear_kernel`. https://github.com/eriklindernoren/ML-From-Scratch/tree/a2806c6732eee8d27762edd6d864e0c179d8e9e8 . Exact source, MIT license, and SHA-256 hashes in `docs/reference/svm/`. Read for dissection; CVXOPT implementation not executed.
- scikit-learn 1.9.1, `SVC`, https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html ; installed `svm/_base.py`, `BaseSVC._get_coef` and `predict`, inspected for linear coefficients and multiclass prediction. Source hashes and software versions in `results/w10_svm/audit.json`. StandardScaler, GroupShuffleSplit, and classification metrics used for the controlled experiment.
- UCI HAR v1.0, https://doi.org/10.24432/C54S4K ; unchanged dataset and original training population. W10 uses the previously defined subject-held-out validation split. No new test-set evaluation.

### W09 HMM -- 2026-10-27

- HCMUT, Faculty of Computer Science and Engineering. Hidden Markov Models, lecture slides dated 11 August 2026. Pages 8, 11-13: model definition and Forward; pages 20-25: Viterbi and backtracking. Local file: `C:/Users/USER/Downloads/ML/HMM.pdf`.
- D. D. Bourgin, numpy-ml, commit `b0359af5285fbf9699d64fd5ec059493228af03e`. `numpy_ml/hmm/hmm.py`: `MultinomialHMM._forward`, `log_likelihood`, and `decode`; `numpy_ml/utils/misc.py`: `logsumexp`. https://github.com/ddbourgin/numpy-ml/tree/b0359af5285fbf9699d64fd5ec059493228af03e . Exact source snapshots, license, and SHA-256 hashes are saved in `docs/reference/numpy_ml_hmm/`.
- Reyes-Ortiz et al., UCI Human Activity Recognition Using Smartphones, v1.0. Dataset README and https://doi.org/10.24432/C54S4K . Source for the subject split, 128-sample windows, and 50% overlap. The experiment reconstructs conservative continuous segments by checking overlap across all nine channels.
- Installed scikit-learn, version recorded in `results/w09_hmm/audit.json`: `StandardScaler`, `MiniBatchKMeans`, `GroupShuffleSplit`, `f1_score`, `accuracy_score`, and `confusion_matrix`, used for preprocessing, quantization, subject separation, and evaluation.

### W05
- Tom Mitchell (1997) *Machine Learning*, Chapter 4 (Artificial Neural Networks), Sections 4.4–4.5 (Perceptron, Delta rule, Gradient descent).
- Christopher M. Bishop (2006) *Pattern Recognition and Machine Learning*, Chapter 4 (Linear Models for Classification), Section 4.1.7 (Perceptron algorithm).

### 2026-10-14 audit
- Installed scikit-learn 1.9.1, `sklearn/naive_bayes.py`, `GaussianNB._partial_fit`: inspected local source for `epsilon_ = var_smoothing * np.var(X, axis=0).max()`; numerical comparison in `experiments/part1_pre_midterm/w06_naive_bayes.py`.
- Installed scikit-learn 1.9.1, `sklearn.metrics.f1_score`, `confusion_matrix`, and `log_loss`: numerical references for six-class evaluation and probability diagnostics. Regression tests are in `tests/test_metrics.py` and `tests/test_naive_bayes.py`.
- Installed scikit-learn 1.9.1, `sklearn.model_selection.GroupShuffleSplit`: subject-disjoint validation from the original training population; implementation and split metadata in `experiments/part1_pre_midterm/common.py` and the result audit JSON files.

### Tree, MLP, GA and BN/TAN practice -- 2026-10-14
- Erik Linder-Noren, ML-From-Scratch, `mlfromscratch/supervised_learning/decision_tree.py`, `DecisionTree._build_tree`, `ClassificationTree._calculate_information_gain`, `_majority_vote`. Snapshot SHA-256 `825d8963080da526683b709cff4aa2f190543c129faaf4696128f6dde74505f2`, MIT; saved with license in `docs/reference/`.
- Erik Linder-Noren, ML-From-Scratch, `mlfromscratch/unsupervised_learning/genetic_algorithm.py`, `_initialize`, `_calculate_fitness`, `_crossover`, `_mutate`, `run`. Snapshot SHA-256 `3f1836628cc5c58032ffb20b37b490ed0b9641ce1eca0afdf321c0411c832ae1`, MIT; saved with license in `docs/reference/`.
- Installed sklearn 1.9.1, `DecisionTreeClassifier(criterion="entropy")`, `ccp_alpha` comparison; `sklearn.feature_selection.f_classif` for the fit-only GA candidate ranking. The exact adaptation and function/line mapping are in `docs/reference/DISSECTION.md`.
- Existing NumPy MLP core, `src/from_scratch/mlp.py`, `init_params`, `forward`, `loss`, `backward`; numerical gradient contract and the new paired-capacity experiment in `w05_mlp.py`. New training wrapper is Codex-assisted, dated 14 Oct.
- BN review: supplied graph and specified CPTs in `experiments/part1_pre_midterm/w07_bn.py`; exact 32-state enumeration checks factorization and the worked conditional-independence cases. This is direct numerical verification, not a claimed external textbook consultation.
