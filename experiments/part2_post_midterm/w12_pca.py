"""W12 PCA checks and HAR experiment (no official test reads).

Run: python -m experiments.part2_post_midterm.w12_pca
Fixed before execution: k={2,5,10,20,50,100,200,561}; linear SVC C=0.01
from W10, no whitening or additional rescaling after PCA. No summary post.
"""
import ast
import csv
import hashlib
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
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.data import load_data, RAW_DIR
from src.from_scratch.pca import fit_pca, transform, inverse_transform
from experiments.part2_post_midterm.w10_svm import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/w12_pca'
K_VALUES = (2, 5, 10, 20, 50, 100, 200, 561)
COMMIT = 'a2806c6732eee8d27762edd6d864e0c179d8e9e8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_out(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reference_checks():
    folder = ROOT / 'docs/reference/pca'
    provenance = json.loads((folder / 'provenance.json').read_text())
    for name, item in provenance['files'].items():
        assert sha(folder / name) == item['sha256'], name
    # Execute exact AST definitions from the pinned files. Inject only np and
    # the upstream covariance helper, avoiding unrelated package imports.
    scope = {'np': np}
    for filename, definition in [('data_operation.py', 'calculate_covariance_matrix'),
                                 ('principal_component_analysis.py', 'PCA')]:
        tree = ast.parse((folder / filename).read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name == definition]
        assert len(nodes) == 1
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5)) @ np.diag([4., 3., 2., 1., .3]) + [10, -4, 2, 7, 3]
    cases = []
    for k in (1, 3, 5):
        mean, W, ev = fit_pca(X, k)
        ref = PCA(n_components=k, svd_solver='full').fit(X)
        projector_error = float(np.max(np.abs(W.T @ W - ref.components_.T @ ref.components_)))
        np.testing.assert_allclose(W.T @ W, ref.components_.T @ ref.components_, atol=1e-10)
        np.testing.assert_allclose(ev, ref.explained_variance_, atol=1e-10)
        own_reconstruction = inverse_transform(transform(X, mean, W), mean, W)
        np.testing.assert_allclose(own_reconstruction, ref.inverse_transform(ref.transform(X)), atol=1e-10)
        # Upstream computes centered covariance, but projects uncentered X.
        raw = np.real_if_close(scope['PCA']().transform(X, k))
        centered = np.real_if_close(scope['PCA']().transform(X-mean, k))
        np.testing.assert_allclose(raw-raw.mean(axis=0), centered, atol=1e-10)
        # Gram matrices are invariant to eigenvector sign/orthogonal rotation.
        own = transform(X, mean, W)
        np.testing.assert_allclose(centered @ centered.T, own @ own.T, atol=1e-9)
        cases.append({'k': k, 'sklearn_projector_max_abs_error': projector_error,
                      'upstream_raw_score_mean_norm': float(np.linalg.norm(raw.mean(axis=0))),
                      'upstream_centered_gram_max_abs_error': float(np.max(np.abs(centered @ centered.T-own @ own.T)))})
    return {'cases': cases, 'source_hashes_verified': True,
            'execution_adapter': 'Exact AST class/helper definitions; injected np and covariance helper; no algorithm edits',
            'upstream_observation': 'Covariance uses centering and n-1; transform projects raw X and refits on each call. Pass centered data for score comparison; do not refit on validation.',
            'equation_map': {'own': 'fit_pca: mean, thin SVD, s^2/(n-1); transform: (X-mean)W^T; inverse_transform: ZW+mean',
                             'upstream': 'PCA.transform: covariance -> eig -> descending eigenvalues -> X.dot(eigenvectors)',
                             'sklearn': 'PCA._fit_full: center -> SVD -> variance -> sign convention -> truncate'}}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = reference_checks()
    X, y, subjects = load_data('train')
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X, y, subjects))
    assert not set(subjects[fit]) & set(subjects[val])
    old = json.loads((ROOT / 'results/w10_svm/audit.json').read_text())
    hashes = {name: sha(RAW_DIR / 'UCI HAR Dataset/train' / name)
              for name in ('X_train.txt', 'y_train.txt', 'subject_train.txt')}
    assert hashes == old['input_sha256']
    assert sorted(map(int, np.unique(subjects[fit]))) == old['fit_subjects']
    assert sorted(map(int, np.unique(subjects[val]))) == old['validation_subjects']
    scaler = StandardScaler().fit(X[fit])
    A, B = scaler.transform(X[fit]), scaler.transform(X[val])
    start = perf_counter()
    mean, W, ev = fit_pca(A, A.shape[1])
    pca_seconds = perf_counter()-start
    ref = PCA(n_components=A.shape[1], svd_solver='full').fit(A)
    np.testing.assert_allclose(ev, ref.explained_variance_, rtol=1e-7, atol=1e-10)
    Za, Zb = transform(A, mean, W), transform(B, mean, W)
    np.testing.assert_allclose(np.cov(Za, rowvar=False), np.diag(ev), atol=1e-9)
    comparisons = []
    for k in K_VALUES:
        rp = ref.components_[:k]
        err = float(np.max(np.abs(W[:k].T @ W[:k]-rp.T @ rp)))
        assert err < 1e-7
        xr = inverse_transform(Zb[:, :k], mean, W[:k])
        xr_ref = (B-ref.mean_) @ rp.T @ rp + ref.mean_
        np.testing.assert_allclose(xr, xr_ref, atol=1e-7)
        comparisons.append({'k': k, 'projector_max_abs_error': err,
                            'validation_reconstruction_max_abs_error': float(np.max(np.abs(xr-xr_ref)))})
    audit = {'date': '2026-11-17', 'test_loaded': False, 'split_seed': 42,
             'fit_subjects': old['fit_subjects'], 'validation_subjects': old['validation_subjects'],
             'fit_rows': len(fit), 'validation_rows': len(val), 'input_sha256': hashes,
             'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
             'source_sha256': {'own_pca': sha(ROOT/'src/from_scratch/pca.py'), 'experiment': sha(__file__),
                               'sklearn_pca': sha(inspect.getsourcefile(PCA))},
             'reference_checks': checks, 'har_sklearn_comparisons': comparisons,
             'pca_fit_seconds': pca_seconds, 'pca_variance_ddof': 1, 'whiten': False,
             'k_grid': K_VALUES, 'fit_variance_k95': int(np.searchsorted(np.cumsum(ev)/ev.sum(), .95)+1),
             'classifier': {'class': 'sklearn.svm.SVC', 'kernel': 'linear', 'C': .01, 'tol': .001,
                            'cache_size_MB': 512, 'decision_function_shape': 'ovo'},
             'selection_rule': 'highest validation Macro-F1 among reduced k<561; smaller k breaks exact ties',
             'limitations': ['Reused validation holdout; no unbiased final estimate',
                            'Single-run timing; deterministic fixed-split fits',
                            'Overlapping windows are correlated',
                            'MSE is averaged over rows and all 561 standardized input features',
                            'One full SVD reused for nested k; PCA time is not measured separately per k',
                            'Historical official-test exposure remains as previously documented'],
             'runs': []}
    predictions, rows = [], []
    for k in (None, *K_VALUES):
        name = 'no_pca' if k is None else f'pca_{k}'
        train, valid = (A, B) if k is None else (Za[:, :k], Zb[:, :k])
        model = SVC(C=.01, kernel='linear', cache_size=512, decision_function_shape='ovo')
        start = perf_counter()
        model.fit(train, y[fit])
        seconds = perf_counter()-start
        assert model.fit_status_ == 0
        start = perf_counter()
        pred = model.predict(valid)
        pred_seconds = perf_counter()-start
        m = metrics(y[val], pred)
        rec_a = A if k is None else inverse_transform(train, mean, W[:k])
        rec_b = B if k is None else inverse_transform(valid, mean, W[:k])
        train_mse, val_mse = float(np.mean((A-rec_a)**2)), float(np.mean((B-rec_b)**2))
        if k is not None:
            np.testing.assert_allclose(train_mse, (len(A)-1)*ev[k:].sum()/(len(A)*A.shape[1]), atol=1e-10)
        record = {'model': name, 'k': k, 'retained_fit_variance': 1. if k is None else float(ev[:k].sum()/ev.sum()),
                  'fit_reconstruction_mse': train_mse, 'validation_reconstruction_mse': val_mse,
                  'validation': m, 'fit_macro_f1': metrics(y[fit], model.predict(train))['macro_f1'],
                  'classifier_fit_seconds': seconds, 'validation_predict_seconds': pred_seconds,
                  'support_vectors': int(model.n_support_.sum()), 'fit_status': int(model.fit_status_)}
        audit['runs'].append(record)
        rows.append({key: value for key, value in record.items() if key != 'validation'} |
                    {'validation_macro_f1': m['macro_f1'], 'validation_accuracy': m['accuracy']})
        for i, label in zip(val, pred):
            predictions.append({'model': name, 'source_row_1based': int(i+1), 'subject': int(subjects[i]),
                                'true': int(y[i]), 'predicted': int(label)})
        if k is None:
            with (ROOT/'results/w10_svm/predictions.csv').open() as f:
                archived = [r for r in csv.DictReader(f) if float(r['C']) == .01]
            np.testing.assert_array_equal([int(r['source_row_1based']) for r in archived], val+1)
            np.testing.assert_array_equal([int(r['predicted']) for r in archived], pred)
            audit['w10_baseline_predictions_match'] = True
        print(f'{name}: variance={record["retained_fit_variance"]:.4f}, val MSE={val_mse:.6f}, F1={m["macro_f1"]:.6f}', flush=True)
    reduced = [r for r in audit['runs'] if r['k'] is not None and r['k'] < 561]
    best = max(reduced, key=lambda r: (r['validation']['macro_f1'], -r['k']))
    audit['selected_reduced_k'] = best['k']
    for field in ('fit_reconstruction_mse', 'validation_reconstruction_mse'):
        assert np.all(np.diff([r[field] for r in audit['runs'][1:]]) <= 1e-10)
    assert audit['runs'][-1]['validation_reconstruction_mse'] < 1e-20
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    csv_out('metrics.csv', rows)
    csv_out('predictions.csv', predictions)
    csv_out('spectrum.csv', [{'component': i+1, 'variance': v, 'variance_ratio': v/ev.sum(),
                              'cumulative_ratio': np.sum(ev[:i+1])/ev.sum()} for i,v in enumerate(ev)])
    runs = audit['runs'][1:]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].semilogx(K_VALUES, [r['retained_fit_variance'] for r in runs], 'o-')
    axes[0].axhline(.95, color='grey', linestyle='--')
    axes[0].set(xlabel='Components k', ylabel='Fraction', title='Retained fit variance')
    for field, label in [('fit_reconstruction_mse', 'Fit'), ('validation_reconstruction_mse', 'Validation')]:
        axes[1].semilogx(K_VALUES, [r[field] for r in runs], 'o-', label=label)
    axes[1].set(xlabel='Components k', ylabel='MSE (standardized units)', title='Reconstruction error')
    axes[1].legend()
    axes[2].semilogx(K_VALUES, [r['validation']['macro_f1'] for r in runs], 'o-', label='PCA + linear SVM')
    axes[2].axhline(audit['runs'][0]['validation']['macro_f1'], linestyle='--', color='grey', label='No PCA')
    axes[2].set(xlabel='Components k', ylabel='Validation Macro-F1', title='Fixed C=0.01')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(OUT/'pca_sensitivity.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
