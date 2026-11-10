"""W11: fixed-grid RBF SVC and same-split comparisons; no official test reads.

Run: python -m experiments.part2_post_midterm.w11_kernel_svm
Grid and comparison settings were specified before the first run on 2026-11-09.
"""
import csv
import hashlib
import importlib.util
import inspect
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.data import ACTIVITY_LABELS, RAW_DIR, load_data
from experiments.part2_post_midterm.w10_svm import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/w11_kernel_svm'
C_VALUES = (0.1, 1., 10.)
GAMMAS = (0.0001, 0.001, 0.01, 0.1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_checks():
    """Execute only the pinned kernel helpers, not the upstream CVXOPT solver."""
    folder = ROOT / 'docs/reference/svm'
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, info in provenance['files'].items():
        assert sha(folder / name) == info['sha256'], name
    spec = importlib.util.spec_from_file_location('pinned_svm_kernels', folder / 'kernels.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    X = np.array([[-2., -1.], [-1., -1.], [-.2, .4], [.3, -.2], [1., 1.], [2., 1.]])
    y = np.array([-1, -1, 1, -1, 1, 1])
    errors = []
    for gamma in GAMMAS:
        K = np.array([[ref.rbf_kernel(gamma)(a, b) for b in X] for a in X])
        np.testing.assert_allclose(K, rbf_kernel(X, gamma=gamma), atol=1e-12)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 1)
        assert np.linalg.eigvalsh(K).min() > -1e-10
        model = SVC(C=.7, kernel='rbf', gamma=gamma, tol=1e-8).fit(X, y)
        manual = K[:, model.support_] @ model.dual_coef_[0] + model.intercept_[0]
        error = float(np.max(np.abs(manual-model.decision_function(X))))
        assert error < 1e-10
        signed = model.dual_coef_[0]
        assert abs(signed.sum()) < 1e-8 and np.all(np.abs(signed) <= .7+1e-8)
        pre = SVC(C=.7, kernel='precomputed', tol=1e-8).fit(K, y)
        np.testing.assert_array_equal(pre.predict(K), model.predict(X))
        np.testing.assert_allclose(pre.decision_function(K), manual, atol=1e-6)
        errors.append(error)
    # Check the lecture's polynomial kernel with an explicit 2D feature map.
    phi = lambda x: np.array([1, np.sqrt(2)*x[0], np.sqrt(2)*x[1],
                             np.sqrt(2)*x[0]*x[1], x[0]**2, x[1]**2])
    for a in X:
        for b in X:
            np.testing.assert_allclose(phi(a) @ phi(b), ref.polynomial_kernel(2, 1)(a, b), atol=1e-12)
    return {'reference_hashes_verified': True, 'kernel_and_feature_map_checks': True,
            'binary_dual_score_max_abs_error': max(errors),
            'precomputed_prediction_parity': True, 'dual_box_and_balance_checks': True}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = kernel_checks()
    X, y, subjects = load_data('train')
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X, y, subjects))
    assert not set(subjects[fit]) & set(subjects[val])
    scaler = StandardScaler().fit(X[fit])
    assert scaler.n_samples_seen_ == len(fit)
    Xfit, Xval = scaler.transform(X[fit]), scaler.transform(X[val])
    hashes = {name: sha(RAW_DIR / 'UCI HAR Dataset/train' / name)
              for name in ('X_train.txt', 'y_train.txt', 'subject_train.txt')}
    old = json.loads((ROOT / 'results/w10_svm/audit.json').read_text())
    assert old['input_sha256'] == hashes
    assert old['fit_subjects'] == sorted(map(int, np.unique(subjects[fit])))
    assert old['validation_subjects'] == sorted(map(int, np.unique(subjects[val])))
    audit = {'date': '2026-11-09', 'test_loaded': False, 'split_seed': 42,
             'fit_subjects': old['fit_subjects'], 'validation_subjects': old['validation_subjects'],
             'fit_windows': len(fit), 'validation_windows': len(val), 'features': X.shape[1],
             'input_sha256': hashes, 'script_sha256': sha(__file__),
             'w10_audit_sha256': sha(ROOT / 'results/w10_svm/audit.json'),
             'sklearn_svc_source_sha256': sha(inspect.getsourcefile(SVC)),
             'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
             'grid': {'C': C_VALUES, 'gamma': GAMMAS},
             'selection': 'largest validation macro-F1; smaller C, then smaller gamma on exact ties',
             'settings': {'tol': .001, 'cache_size_MB': 512, 'decision_function_shape': 'ovo',
                          'class_weight': None, 'probability': False, 'shrinking': True, 'max_iter': -1},
             'checks': checks, 'runs': []}
    predictions = []

    def evaluate(name, model, C='', gamma=''):
        start = perf_counter()
        model.fit(Xfit, y[fit])
        fit_seconds = perf_counter()-start
        if isinstance(model, SVC):
            assert model.fit_status_ == 0
        train = metrics(y[fit], model.predict(Xfit))
        start = perf_counter()
        pred = model.predict(Xval)
        seconds = perf_counter()-start
        r = {'model': name, 'C': C, 'gamma': gamma, 'train': train,
             'validation': metrics(y[val], pred), 'fit_seconds': fit_seconds,
             'validation_predict_seconds': seconds,
             'support_vectors': int(model.n_support_.sum()) if isinstance(model, SVC) else None}
        if isinstance(model, SVC):
            r['iterations_by_pair'] = model.n_iter_.tolist()
            r['fit_status'] = int(model.fit_status_)
            r['core_model_array_bytes'] = sum(a.nbytes for a in
                (model.support_vectors_, model.dual_coef_, model.intercept_, model.support_))
        audit['runs'].append(r)
        for i, label in zip(val, pred):
            predictions.append({'model': name, 'C': C, 'gamma': gamma,
                                'source_row_1based': int(i+1), 'subject': int(subjects[i]),
                                'true': int(y[i]), 'predicted': int(label)})
        print(f'{name} C={C} gamma={gamma}: val F1={r["validation"]["macro_f1"]:.6f}', flush=True)
        return r, pred

    for C in C_VALUES:
        for gamma in GAMMAS:
            evaluate('RBF SVM', SVC(C=C, gamma=gamma, kernel='rbf', cache_size=512,
                                   decision_function_shape='ovo'), C, gamma)
    best = max(audit['runs'], key=lambda r: (r['validation']['macro_f1'], -r['C'], -r['gamma']))
    audit['selected'] = {'C': best['C'], 'gamma': best['gamma']}
    linear, linear_pred = evaluate('Linear SVM (W10 selection)',
        SVC(C=old['selected_C'], kernel='linear', cache_size=512, decision_function_shape='ovo'), old['selected_C'])
    with (ROOT / 'results/w10_svm/predictions.csv').open() as f:
        old_pred = [r for r in csv.DictReader(f) if float(r['C']) == old['selected_C']]
    np.testing.assert_array_equal([int(r['source_row_1based']) for r in old_pred], val+1)
    np.testing.assert_array_equal([int(r['predicted']) for r in old_pred], linear_pred)
    audit['checks']['w10_linear_predictions_reproduced'] = True
    nb, _ = evaluate('GaussianNB (fixed smoothing)', GaussianNB(var_smoothing=1e-9))
    dummy, _ = evaluate('Majority baseline', DummyClassifier(strategy='most_frequent'))
    audit['comparison_settings'] = {'GaussianNB': {'var_smoothing': 1e-9},
                                   'DummyClassifier': {'strategy': 'most_frequent'},
                                   'linear_C': old['selected_C']}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    flat = [{'model': r['model'], 'C': r['C'], 'gamma': r['gamma'],
             'train_macro_f1': r['train']['macro_f1'], 'validation_macro_f1': r['validation']['macro_f1'],
             'validation_accuracy': r['validation']['accuracy'], 'support_vectors': r['support_vectors'],
             'fit_seconds': r['fit_seconds'], 'validation_predict_seconds': r['validation_predict_seconds']}
            for r in audit['runs']]
    write_csv(OUT / 'metrics.csv', flat)
    write_csv(OUT / 'predictions.csv', predictions)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, split, title in zip(axes, ('train', 'validation'), ('Fit Macro-F1', 'Validation Macro-F1')):
        values = np.array([r[split]['macro_f1'] for r in audit['runs'][:12]]).reshape(3, 4)
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set(xticks=range(4), xticklabels=[f'{g:g}' for g in GAMMAS],
               yticks=range(3), yticklabels=[f'{c:g}' for c in C_VALUES], xlabel='gamma', ylabel='C', title=title)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', color='black' if values[i,j]>.65 else 'white')
    fig.colorbar(im, ax=axes, label='Macro-F1')
    fig.savefig(OUT / 'c_gamma_sensitivity.png', dpi=160)
    plt.close(fig)
    lines = ['# W11 - Kernel SVM on HAR', '', 'Run: `python -m experiments.part2_post_midterm.w11_kernel_svm`', '',
             '2026-11-09. Fixed grid specified before execution: C={0.1,1,10}, gamma={0.0001,0.001,0.01,0.1}.', '',
             'Same 5,270 fit / 2,082 validation windows and 15/6 disjoint subjects as W10. StandardScaler fits only the fit rows; all 561 features remain. Official test files are not read. Probability calibration is disabled; SVC uses 15 one-versus-one classifiers.', '',
             '## RBF sensitivity', '', '| C | gamma | Fit F1 | Validation F1 | Accuracy | SV |', '|---:|---:|---:|---:|---:|---:|']
    for r in audit['runs'][:12]:
        lines.append(f'| {r["C"]:g} | {r["gamma"]:g} | {r["train"]["macro_f1"]:.6f} | {r["validation"]["macro_f1"]:.6f} | {r["validation"]["accuracy"]:.6f} | {r["support_vectors"]} |')
    lines += ['', '![C/gamma sensitivity](c_gamma_sensitivity.png)', '',
              f'Selected RBF: C={best["C"]:g}, gamma={best["gamma"]:g}; highest validation Macro-F1, with smaller C then gamma breaking exact ties.', '',
              '## Same-split comparison', '', '| Model | Validation F1 | Accuracy | Fit seconds | Predict seconds |', '|---|---:|---:|---:|---:|']
    for r in (best, linear, nb, dummy):
        lines.append(f'| {r["model"]} | {r["validation"]["macro_f1"]:.6f} | {r["validation"]["accuracy"]:.6f} | {r["fit_seconds"]:.3f} | {r["validation_predict_seconds"]:.3f} |')
    lines += ['', 'GaussianNB is a fresh sklearn baseline with fixed var_smoothing=1e-9, not a reproduction of all W06 tuning. Majority class is learned from fit labels. Linear C is the W10 selection; its validation predictions are reproduced exactly. RBF has 12 candidates versus five W10 linear candidates, so tuning budgets differ. HMM uses sequence context and is not included in this independent-window ranking.', '',
              '## Selected RBF errors', '', 'Rows=true, columns=predicted; labels 1-6 follow WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING.', '',
              '```text', np.array2string(np.array(best['validation']['confusion_matrix'])), '```', '',
              'First five errors in source-row order:', '', '| Row | Subject | True | Predicted |', '|---:|---:|---|---|']
    errors = [r for r in predictions if r['model']=='RBF SVM' and r['C']==best['C'] and r['gamma']==best['gamma'] and r['true']!=r['predicted']][:5]
    for r in errors:
        lines.append(f'| {r["source_row_1based"]} | {r["subject"]} | {ACTIVITY_LABELS[r["true"]]} | {ACTIVITY_LABELS[r["predicted"]]} |')
    lines += ['', '## Verification and limits', '',
              'Pinned reference hashes pass. Executed upstream RBF helpers agree with sklearn pairwise kernels. Small Gram matrices are symmetric, have unit diagonal, and pass a numerical PSD check; this is a check of these matrices, not a general kernel proof. An explicit polynomial feature map matches the lecture kernel. Binary dual-score reconstruction, alpha box/equality constraints, and precomputed-kernel parity pass for all four gamma values. The upstream CVXOPT solver was not executed.', '',
              'All HAR SVC fits converge. This is deterministic fitting on one fixed split, so repeated training seeds are not used. The protocol seed list is not treated as five independent datasets. Timings are single-run wall-clock values with numerical thread pools limited to one; core model array bytes are not total process memory.', '',
              'Results are validation-selected and reuse a holdout examined in earlier weeks; they are not unbiased final-test estimates. Overlapping windows are correlated. RBF distances depend on scaling. High gamma localizes similarity and can overfit; higher C need not improve validation performance. No significance or universal model-ranking claim is made. The historical test exposure recorded in the project remains unchanged.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
