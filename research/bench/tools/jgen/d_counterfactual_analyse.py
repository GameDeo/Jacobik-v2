"""Analyse the preregistered D-B fixed-row Q6 counterfactual."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/D/counterfactual')
ARMS = {'control': ROOT / 'control', 'q6_physical_all': ROOT / 'q6_physical_all'}
RNG = np.random.default_rng(20261104)
NBOOT = 2000


def load_arm(path):
    out = {}
    for f in sorted(path.glob('*.json')):
        rec = json.loads(f.read_text())
        if rec.get('status') == 'ok':
            out[rec['code']] = rec
    return out


arms = {k: load_arm(v) for k, v in ARMS.items()}
codes = sorted(set(arms['control']) & set(arms['q6_physical_all']))
metrics = {arm: {'rho': [], 'regret': [], 'lenient': []} for arm in ARMS}
fail = []
for code in codes:
    row_metrics = {arm: {'rho': [], 'regret': [], 'lenient': []} for arm in ARMS}
    by_seed = {arm: {s['seed']: s for s in arms[arm][code]['seeds']} for arm in ARMS}
    for seed in sorted(set(by_seed['control']) & set(by_seed['q6_physical_all'])):
        by_idx = {}
        for arm in ARMS:
            by_idx[arm] = {r['i']: r for r in by_seed[arm][seed]['rows']
                           if np.isfinite(r.get('E', np.nan)) and np.isfinite(r.get('s_re', np.nan))}
        common = sorted(set(by_idx['control']) & set(by_idx['q6_physical_all']))
        if len(common) < 2:
            continue
        energy = np.asarray([by_idx['control'][i]['E'] for i in common], float)
        floor = float(np.min(energy))
        for arm in ARMS:
            score = np.asarray([by_idx[arm][i]['s_re'] for i in common], float)
            rho = float(spearmanr(score, -energy).statistic)
            top = int(np.argmax(score))
            regret = float(energy[top] - floor)
            lenient = int(np.sum((score >= np.quantile(score, .75)) & ((energy - floor) > 100.0)))
            row_metrics[arm]['rho'].append(rho)
            row_metrics[arm]['regret'].append(regret)
            row_metrics[arm]['lenient'].append(lenient)
    for arm in ARMS:
        vals = row_metrics[arm]
        if vals['rho']:
            for met in vals:
                metrics[arm][met].append(float(np.mean(vals[met])))
    if not row_metrics['control']['rho']:
        fail.append((code, 'no_seed_with_two_finite_rows'))

def summarize(v):
    a = np.asarray(v, float)
    if not len(a): return {'n': 0, 'mean': None, 'ci95': [None, None]}
    b = np.mean(RNG.choice(a, (NBOOT, len(a)), replace=True), axis=1)
    return {'n': int(len(a)), 'mean': float(np.mean(a)), 'ci95': [float(x) for x in np.quantile(b, [.025, .975])]}

summary = {'contract': 'D_COUNTERFACTUAL_CONTRACT.md', 'parents_matched': len(codes), 'failures': fail,
           'arms': {a: {m: summarize(v) for m, v in data.items()} for a, data in metrics.items()}, 'paired_differences': {}}
for m in ('rho', 'regret', 'lenient'):
    a, b = metrics['q6_physical_all'][m], metrics['control'][m]
    dif = np.asarray(a, float) - np.asarray(b, float)
    boot = np.mean(RNG.choice(dif, (NBOOT, len(dif)), replace=True), axis=1) if len(dif) else []
    summary['paired_differences'][m] = {'mean': float(np.mean(dif)) if len(dif) else None,
        'ci95': [float(x) for x in np.quantile(boot, [.025, .975])] if len(dif) else [None, None]}
out = ROOT / 'summary.json'
out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
