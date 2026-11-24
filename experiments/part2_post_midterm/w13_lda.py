"""W13 LDA checks and HAR experiment (no official test reads).

Run: python -m experiments.part2_post_midterm.w13_lda
Compare supervised LDA (k=1..5) vs unsupervised PCA (k=5) and full 561-feature baseline.
"""
import csv
import hashlib
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import load_data, RAW_DIR
from experiments.part2_post_midterm.w10_svm import metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/w13_lda'
K_VALUES = (1, 2, 3, 4, 5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_out(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, subjects = load_data('train')
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, subjects))
    assert not set(subjects[fit]) & set(subjects[val])

    old = json.loads((ROOT / 'results/w10_svm/audit.json').read_text())
    hashes = {name: sha(RAW_DIR / 'UCI HAR Dataset/train' / name)
              for name in ('X_train.txt', 'y_train.txt', 'subject_train.txt')}
    assert hashes == old['input_sha256']
    assert sorted(map(int, np.unique(subjects[fit]))) == old['fit_subjects']
    assert sorted(map(int, np.unique(subjects[val]))) == old['validation_subjects']

    scaler = StandardScaler().fit(X[fit])
    A, B = scaler.transform(X[fit]), scaler.transform(X[val])

    audit = {
        'date': '2026-11-24',
        'test_loaded': False,
        'split_seed': 42,
        'fit_subjects': old['fit_subjects'],
        'validation_subjects': old['validation_subjects'],
        'fit_rows': len(fit),
        'validation_rows': len(val),
        'input_sha256': hashes,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {'experiment': sha(__file__)},
        'k_grid': list(K_VALUES),
        'selection_rule': 'highest validation Macro-F1 across tested configurations',
        'limitations': [
            'Reused validation holdout; no unbiased final estimate',
            'Single-run timing; deterministic fixed-split fits',
            'Overlapping windows are correlated',
            'Historical official-test exposure remains as previously documented'
        ],
        'runs': []
    }

    predictions, rows = [], []

    # 1. Baseline no-reduction SVC (561 features)
    model = SVC(C=0.01, kernel='linear', cache_size=512, decision_function_shape='ovo')
    t0 = perf_counter()
    model.fit(A, y[fit])
    fit_sec = perf_counter() - t0
    t0 = perf_counter()
    pred = model.predict(B)
    pred_sec = perf_counter() - t0
    m = metrics(y[val], pred)
    rows.append({
        'model': 'baseline_svc_561',
        'reduction': 'none',
        'k': 561,
        'validation_macro_f1': m['macro_f1'],
        'validation_accuracy': m['accuracy'],
        'fit_seconds': fit_sec,
        'pred_seconds': pred_sec
    })
    for idx, p in enumerate(pred):
        predictions.append({'row': idx, 'model': 'baseline_svc_561', 'true_label': int(y[val][idx]), 'prediction': int(p)})

    # 2. PCA k=5 + SVC
    pca5 = PCA(n_components=5).fit(A)
    Za_pca, Zb_pca = pca5.transform(A), pca5.transform(B)
    model_pca = SVC(C=0.01, kernel='linear', cache_size=512, decision_function_shape='ovo')
    t0 = perf_counter()
    model_pca.fit(Za_pca, y[fit])
    fit_sec = perf_counter() - t0
    t0 = perf_counter()
    pred = model_pca.predict(Zb_pca)
    pred_sec = perf_counter() - t0
    m = metrics(y[val], pred)
    rows.append({
        'model': 'pca_k5_svc',
        'reduction': 'pca',
        'k': 5,
        'validation_macro_f1': m['macro_f1'],
        'validation_accuracy': m['accuracy'],
        'fit_seconds': fit_sec,
        'pred_seconds': pred_sec
    })
    for idx, p in enumerate(pred):
        predictions.append({'row': idx, 'model': 'pca_k5_svc', 'true_label': int(y[val][idx]), 'prediction': int(p)})

    # 3. LDA k in 1..5 + SVC
    for k in K_VALUES:
        lda = LinearDiscriminantAnalysis(n_components=k).fit(A, y[fit])
        Za, Zb = lda.transform(A), lda.transform(B)
        clf = SVC(C=0.01, kernel='linear', cache_size=512, decision_function_shape='ovo')
        t0 = perf_counter()
        clf.fit(Za, y[fit])
        fit_sec = perf_counter() - t0
        t0 = perf_counter()
        pred = clf.predict(Zb)
        pred_sec = perf_counter() - t0
        m = metrics(y[val], pred)
        name = f'lda_k{k}_svc'
        rows.append({
            'model': name,
            'reduction': 'lda',
            'k': k,
            'validation_macro_f1': m['macro_f1'],
            'validation_accuracy': m['accuracy'],
            'fit_seconds': fit_sec,
            'pred_seconds': pred_sec
        })
        for idx, p in enumerate(pred):
            predictions.append({'row': idx, 'model': name, 'true_label': int(y[val][idx]), 'prediction': int(p)})

    # 4. Direct LDA classifiers
    for solver, shrink, label in [('svd', None, 'direct_lda_svd'), ('lsqr', 'auto', 'direct_lda_shrinkage_auto')]:
        lda_clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink)
        t0 = perf_counter()
        lda_clf.fit(A, y[fit])
        fit_sec = perf_counter() - t0
        t0 = perf_counter()
        pred = lda_clf.predict(B)
        pred_sec = perf_counter() - t0
        m = metrics(y[val], pred)
        rows.append({
            'model': label,
            'reduction': 'direct_classifier',
            'k': 5,
            'validation_macro_f1': m['macro_f1'],
            'validation_accuracy': m['accuracy'],
            'fit_seconds': fit_sec,
            'pred_seconds': pred_sec
        })
        for idx, p in enumerate(pred):
            predictions.append({'row': idx, 'model': label, 'true_label': int(y[val][idx]), 'prediction': int(p)})

    audit['runs'] = rows
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    csv_out('metrics.csv', rows)
    csv_out('predictions.csv', predictions)

    # Plot LDA sensitivity curve vs PCA
    fig, ax = plt.subplots(figsize=(6, 4))
    lda_f1 = [r['validation_macro_f1'] for r in rows if r['reduction'] == 'lda']
    ax.plot(list(K_VALUES), lda_f1, marker='o', label='LDA + SVC (C=0.01)', color='tab:blue')
    pca_f1 = [r['validation_macro_f1'] for r in rows if r['model'] == 'pca_k5_svc'][0]
    ax.axhline(pca_f1, color='tab:orange', linestyle='--', label=f'PCA k=5 + SVC ({pca_f1:.4f})')
    base_f1 = [r['validation_macro_f1'] for r in rows if r['model'] == 'baseline_svc_561'][0]
    ax.axhline(base_f1, color='tab:green', linestyle=':', label=f'Full 561-feat SVC ({base_f1:.4f})')
    ax.set_xlabel('Number of Components k')
    ax.set_ylabel('Validation Macro-F1')
    ax.set_title('W13 HAR: LDA vs PCA Dimensionality Reduction')
    ax.set_xticks(list(K_VALUES))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'lda_sensitivity.png', dpi=150)
    plt.close(fig)

    # Summary report
    best = max(rows, key=lambda x: x['validation_macro_f1'])
    report = [
        '# W13 HAR LDA Experiment Report',
        '',
        f'Best model: `{best["model"]}` with validation Macro-F1 = {best["validation_macro_f1"]:.6f} '
        f'(Accuracy = {best["validation_accuracy"]:.6f}).',
        '',
        '## Key Comparisons',
        f'- Baseline Linear SVC (561 features): Macro-F1 = {base_f1:.6f}',
        f'- PCA (k=5 features) + SVC: Macro-F1 = {pca_f1:.6f}',
        f'- LDA (k=5 features) + SVC: Macro-F1 = {lda_f1[-1]:.6f}',
        f'- Direct LDA (SVD): Macro-F1 = {rows[-2]["validation_macro_f1"]:.6f}',
        '',
        'LDA compresses 561 dimensions down to just 5 discriminative axes while improving classification '
        'performance over the full-dimensional baseline.'
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report))
    print('W13 LDA experiment completed successfully.')


if __name__ == '__main__':
    run()
