"""eta2 face-plane package, Step 0: crystal survey of eta2 faces (geometry only).

Reads the stored QO observation rows (tmQM X1-X3 + LnQM; raw face/substituent coordinates relative to the metal,
no QM columns) and compares, per eta2 face:
  old normal: metal direction with its bond component removed (production orientation.haptic_features);
  new normal: plane containing the A=B bond that best fits the substituent bond unit vectors (bond component
              removed, uncentred; closed form), sign from the metal.
Reports, per substituent-count pattern (subs per face atom, sorted): row counts, new tilt spread, per-substituent
bend spread (H / X) under both normals, the share of crystal H bent > +20 deg toward the metal, and the in-plane
fit residual. No source change; output JSON under research/data/jgen/campaign/eta2/.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OBS = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/archive/data/refit/observations')
SPLITS = ('X1', 'X2', 'X3', 'LNf')
H_MAX = 1.25  # orientation.PER_SUB_H_MAX
S20 = np.sin(np.radians(20))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def plane_normal(b, us):
    """Normal of the plane through the bond b best fitting the substituent unit vectors us (bond part removed)."""
    e1 = unit(np.cross(b, [1.0, 0, 0]) if abs(b[0]) < 0.9 else np.cross(b, [0, 1.0, 0]))
    e2 = np.cross(b, e1)
    P = np.array([[u @ e1, u @ e2] for u in us])
    sxx, syy, sxy = (P[:, 0] ** 2).sum(), (P[:, 1] ** 2).sum(), (P[:, 0] * P[:, 1]).sum()
    th = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    d = np.cos(th) * e1 + np.sin(th) * e2              # in-plane direction (principal axis)
    lam = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]]))
    return np.cross(b, d), float(lam[0] / max(lam.sum(), 1e-12))


def q(x, ps=(1, 5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, float)
    return {'n': int(len(x)), **({f'p{p}': round(float(np.percentile(x, p)), 3) for p in ps} if len(x) else {})}


def main():
    acc = defaultdict(lambda: defaultdict(list))
    for s in SPLITS:
        df = pd.read_parquet(OBS / f'observations_QO_{s}.parquet',
                             columns=['csd_code', 'k', 'sub_elems', 'raw_face', 'raw_subs'])
        df = df[df.k == 12]
        for r in df.itertuples():
            F = np.asarray(r.raw_face, float).reshape(-1, 3)
            S = np.asarray(r.raw_subs, float).reshape(-1, 4)
            c = F.mean(0)
            uM, b = unit(-c), unit(F[1] - F[0])
            if uM is None or b is None:
                continue
            cnt = [int((S[:, 0] == i).sum()) for i in (0, 1)] if len(S) else [0, 0]
            pat = '%d%d' % tuple(sorted(cnt, reverse=True))
            a = acc[pat]
            a['_src'].append(s)
            n_old = unit(uM - (uM @ b) * b)
            vs = [(s_[1:] - F[int(s_[0])]) for s_ in S]
            us = [unit(v) for v in vs]
            us_ok = [(u, np.linalg.norm(v)) for u, v in zip(us, vs) if u is not None]
            perp = [u - (u @ b) * b for u, _ in us_ok]
            if not us_ok or max(np.linalg.norm(p) for p in perp) < 1e-3:
                a['degenerate'].append(1)
                continue
            n_new, resid = plane_normal(b, [u for u, _ in us_ok])
            if n_new @ uM < 0:
                n_new = -n_new
            a['tilt_new'].append(float(n_new @ uM))
            a['resid'].append(resid)
            hb = []
            for (u, L), s_ in zip(us_ok, S):
                cls = 'H' if L < H_MAX else 'X'
                a[f'bend_old_{cls}'].append(float(u @ n_old))
                a[f'bend_new_{cls}'].append(float(u @ n_new))
                if cls == 'H':
                    hb.append((int(s_[0]), float(u @ n_new)))
            # geminal H pairs: both H on one face atom; record whether they sit on opposite sides of the new plane
            for fa in (0, 1):
                g = [v for i, v in hb if i == fa]
                if len(g) == 2:
                    a['gem_sum'].append(g[0] + g[1])
                    a['gem_opposite'].append(int(g[0] * g[1] < 0))
    out = {}
    for pat, a in sorted(acc.items(), key=lambda kv: -len(kv[1]['_src'])):
        o = {'rows': len(a['_src']), 'by_source': {s: a['_src'].count(s) for s in SPLITS},
             'degenerate': len(a['degenerate']), 'tilt_new': q(a['tilt_new']), 'plane_resid': q(a['resid'])}
        for k in ('bend_old_H', 'bend_new_H', 'bend_old_X', 'bend_new_X'):
            o[k] = q(a[k])
        for cls in ('H', 'X'):
            bn = np.asarray(a[f'bend_new_{cls}'])
            o[f'frac_{cls}_toward_gt20_new'] = round(float((bn > S20).mean()), 4) if len(bn) else None
            o[f'frac_{cls}_toward_gt20_old'] = round(float((np.asarray(a[f'bend_old_{cls}']) > S20).mean()), 4) \
                if len(bn) else None
        if a['gem_sum']:
            o['geminal_H'] = {'pairs': len(a['gem_sum']), 'frac_opposite_sides': round(float(np.mean(a['gem_opposite'])), 4),
                              'bend_sum': q(a['gem_sum'])}
        out[pat] = o
    target = Path('research/data/jgen/campaign/eta2')
    target.mkdir(parents=True, exist_ok=True)
    (target / 'step0_survey.json').write_text(json.dumps(out, indent=1) + '\n')
    for pat, o in out.items():
        print(pat, o['rows'], 'tilt_new', o['tilt_new'].get('p5'), o['tilt_new'].get('p50'),
              'H>20 new/old', o['frac_H_toward_gt20_new'], o['frac_H_toward_gt20_old'],
              'X>20 new/old', o['frac_X_toward_gt20_new'], o['frac_X_toward_gt20_old'])


if __name__ == '__main__':
    main()
