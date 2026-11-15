"""Parent-clustered analysis for the registered D-B Q7 rescoring arms."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/D/counterfactual')
ARMS = {
    'control': ROOT / 'control',
    'q7_bound_centre': ROOT / 'q7_bound_centre',
    'q6_q7_all': ROOT / 'q6_q7_all',
}
RNG = np.random.default_rng(20261104)
NBOOT = 2000

def load(path):
    rows = {}
    for file in sorted(path.glob('*.json')):
        try:
            rec = json.loads(file.read_text())
        except Exception:
            continue
        if rec.get('status') == 'ok':
            rows[rec['code']] = rec
    return rows

data = {arm: load(path) for arm, path in ARMS.items()}
codes = sorted(set.intersection(*(set(v) for v in data.values())))
per_parent = {arm: {} for arm in ARMS}
exclusions = []
for code in codes:
    seeds = {arm: {int(s['seed']): s for s in data[arm][code]['seeds']} for arm in ARMS}
    common_seeds = sorted(set.intersection(*(set(v) for v in seeds.values())))
    values = {arm: {'rho': [], 'regret': [], 'lenient': []} for arm in ARMS}
    used_seeds = 0
    for seed in common_seeds:
        by_arm = {}
        for arm in ARMS:
            by_arm[arm] = {int(r['i']): r for r in seeds[arm][seed]['rows']
                           if np.isfinite(r.get('E', np.nan)) and np.isfinite(r.get('s_re', np.nan))}
        common_idx = sorted(set.intersection(*(set(v) for v in by_arm.values())))
        if len(common_idx) < 2:
            continue
        energy = np.asarray([by_arm['control'][i]['E'] for i in common_idx], float)
        if any(abs(float(by_arm[arm][i]['E']) - energy[j]) > 1e-6
               for arm in ARMS for j, i in enumerate(common_idx)):
            exclusions.append({'code': code, 'seed': seed, 'reason': 'stored_energy_mismatch'})
            continue
        floor = float(np.min(energy))
        for arm in ARMS:
            score = np.asarray([by_arm[arm][i]['s_re'] for i in common_idx], float)
            values[arm]['rho'].append(float(spearmanr(score, -energy).statistic))
            values[arm]['regret'].append(float(energy[int(np.argmax(score))] - floor))
            values[arm]['lenient'].append(int(np.sum((score >= np.quantile(score, .75)) & ((energy - floor) > 100.0))))
        used_seeds += 1
    if not used_seeds:
        exclusions.append({'code': code, 'reason': 'no_common_seed_with_two_rows'})
        continue
    for arm in ARMS:
        per_parent[arm][code] = {metric: float(np.mean(vals)) for metric, vals in values[arm].items()}

common_codes = sorted(set.intersection(*(set(v) for v in per_parent.values())))
def bootstrap(vals):
    a = np.asarray(vals, float)
    if len(a) == 0:
        return {'n': 0, 'mean': None, 'ci95': [None, None]}
    bs = np.mean(RNG.choice(a, (NBOOT, len(a)), replace=True), axis=1)
    return {'n': int(len(a)), 'mean': float(a.mean()), 'ci95': [float(x) for x in np.quantile(bs, [.025, .975])]}

summary = {'contract': 'D_COUNTERFACTUAL_Q7_CONTRACT.md', 'parents_matched': len(common_codes),
           'parents_by_arm': {arm: len(v) for arm, v in per_parent.items()}, 'exclusions': exclusions,
           'arms': {}, 'paired_differences': {}}
for arm in ARMS:
    summary['arms'][arm] = {metric: bootstrap([per_parent[arm][c][metric] for c in common_codes])
                            for metric in ('rho', 'regret', 'lenient')}
    summary['arms'][arm]['lenient_parents'] = sum(
        per_parent[arm][c]['lenient'] > 0 for c in common_codes)
for arm in ('q7_bound_centre', 'q6_q7_all'):
    summary['paired_differences'][arm] = {}
    for metric in ('rho', 'regret', 'lenient'):
        delta = [per_parent[arm][c][metric] - per_parent['control'][c][metric] for c in common_codes]
        summary['paired_differences'][arm][metric] = bootstrap(delta)

out = ROOT / 'q7_summary.json'
out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
