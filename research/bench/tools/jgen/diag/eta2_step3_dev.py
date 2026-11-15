"""eta2 face-plane package, Step 3: summarise the dev arms (top-1 / floor dE per parent and seed, and the smallest
geminal H-C-H angle over the pool at metal-bound CH2 carbons, from the exported pool XYZ; element + distance only)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('research/data/jgen/campaign/eta2/dev')
METALS = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
          'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'La', 'Lu'}


def geminal_min(path):
    lines = path.read_text().splitlines()
    el = [ln.split()[0] for ln in lines[2:]]
    X = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:]])
    m = next(i for i, e in enumerate(el) if e in METALS)
    best = 180.0
    for c in (i for i, e in enumerate(el) if e == 'C' and np.linalg.norm(X[i] - X[m]) < 2.8):
        d = np.linalg.norm(X - X[c], axis=1)
        hs = [i for i, e in enumerate(el) if e == 'H' and d[i] < 1.25]
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                u, v = X[hs[a]] - X[c], X[hs[b]] - X[c]
                best = min(best, float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))))
    return best, json.loads(lines[1])


def main(arms):
    out = {}
    for arm in arms:
        for rec in sorted((ROOT / arm / 'records').glob('*.json')):
            r = json.loads(rec.read_text())
            code, seed = r['code'], r['seed']
            pools = sorted((ROOT / arm / 'xyz' / code).glob(f'*s{seed}_row*.xyz'))
            g = [geminal_min(p) for p in pools]
            top = [a for a, h in g if h.get('is_top1')]
            out.setdefault(code, {}).setdefault(arm, {})[seed] = dict(
                top1_dE=r['top1_dE'], floor_dE=r['floor_dE'], status=r['status'], n_rows=r['n_rows'],
                fp_diff=r['fingerprint_fields_differing'], db=r['constraints_db_sha256'][:8],
                min_hch_pool=round(min((a for a, _ in g), default=180.0), 1),
                n_lt15=sum(a < 15 for a, _ in g), n_lt60=sum(a < 60 for a, _ in g),
                min_hch_top1=round(top[0], 1) if top else None)
    for code, arms_ in out.items():
        print(code)
        for arm, seeds in arms_.items():
            print('  %-6s' % arm, ' | '.join('s%d top1 %7.1f floor %7.1f HCH min %5.1f top1 %s <15:%d <60:%d' % (
                s, v['top1_dE'], v['floor_dE'], v['min_hch_pool'], v['min_hch_top1'], v['n_lt15'], v['n_lt60'])
                for s, v in sorted(seeds.items())))
    (ROOT / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or ['ctl', 'cap', 'pl', 'cappl'])
