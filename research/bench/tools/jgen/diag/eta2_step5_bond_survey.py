"""eta2 face-plane package, Step 5 (owner 2026-11-01): metal-scaled bond test for haptic faces (geometry only).

For every haptic face row in the stored QO observations (tmQM X1-X3 + LnQM; raw face coordinates relative to the
metal, no QM columns) compute r = max_i |M-C_i| / (r_cov(M) + r_cov(C)) with the loader's own radii
(jacobik.topology.mapping._COVALENT_RADII), and report its distribution per hapticity, so the bond/contact cut can be
placed at the gap. Then join the Q1 'C_haptic' rows (csd_code + centroid distance) to their face, to show what the
Q1 C_haptic tables are made of (hapticity, contact share), overall and for the NUWFAM key family (Zr).
Output: research/data/jgen/campaign/eta2/step5_bond_survey.json.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jacobik.core.chemistry import covalent_radius as _covalent_radius

OBS = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/archive/data/refit/observations')
SPLITS = ('X1', 'X2', 'X3', 'LNf')
OUT = Path('research/data/jgen/campaign/eta2/step5_bond_survey.json')
EDGES = np.round(np.arange(0.80, 1.80, 0.025), 3)


def faces():
    rows = []
    for s in SPLITS:
        df = pd.read_parquet(OBS / f'observations_QO_{s}.parquet', columns=['csd_code', 'k', 'metal', 'raw_face'])
        df = df[df.raw_face.notna()]
        for r in df.itertuples():
            F = np.asarray(r.raw_face, float).reshape(-1, 3)
            d = np.linalg.norm(F, axis=1)
            rs = _covalent_radius(r.metal) + _covalent_radius('C')
            rows.append(dict(src=s, csd_code=r.csd_code, metal=r.metal, eta=len(F), cent=float(np.linalg.norm(F.mean(0))),
                             dmax=float(d.max()), dmin=float(d.min()), ratio=float(d.max() / rs),
                             ratio_min=float(d.min() / rs)))
    return pd.DataFrame(rows)


def q1_haptic():
    out = []
    for s in SPLITS:
        df = pd.read_parquet(OBS / f'observations_Q1_{s}.parquet', columns=['csd_code', 'key', 'value'])
        df = df[df.key.str.contains("'C_haptic'")]
        out.append(df.assign(src=s))
    return pd.concat(out, ignore_index=True)


def hist(x):
    h, _ = np.histogram(np.clip(x, EDGES[0], EDGES[-1] - 1e-9), EDGES)
    return {f'{a:.3f}': int(c) for a, c in zip(EDGES[:-1], h)}


def q(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    return {'n': int(len(x)), **({f'p{p}': round(float(np.percentile(x, p)), 3) for p in ps} if len(x) else {})}


def main():
    F = faces()
    res = {'faces_by_eta': {int(e): int(n) for e, n in F.eta.value_counts().sort_index().items()}}
    res['ratio_max_by_eta'] = {int(e): q(g.ratio) for e, g in F.groupby('eta')}
    res['ratio_max_hist_eta2'] = hist(F[F.eta == 2].ratio)
    res['ratio_max_hist_eta5'] = hist(F[F.eta == 5].ratio)
    # the absolute 2.6 A cut of DB v2, per metal radius class: share of eta2 faces it drops
    e2 = F[F.eta == 2].assign(rsum=lambda d: d.dmax / d.ratio)
    res['abs26_drop_by_rsum'] = {f'{lo:.2f}-{lo + 0.2:.2f}': dict(n=int(len(g)), drop=round(float((g.dmax >= 2.6).mean()), 3),
                                                                    ratio_p50=round(float(g.ratio.median()), 3))
                                 for lo in np.arange(2.0, 3.0, 0.2)
                                 for g in [e2[(e2.rsum >= lo) & (e2.rsum < lo + 0.2)]] if len(g)}
    # join Q1 C_haptic rows to their face by csd_code + centroid distance
    Q = q1_haptic()
    Q['metal'] = Q.key.map(lambda k: ast.literal_eval(k)[0])
    m = Q.merge(F[['csd_code', 'metal', 'eta', 'cent', 'ratio']], on=['csd_code', 'metal'], how='left')
    m = m[(m.cent - m.value).abs() < 2e-3].drop_duplicates(subset=['src', 'csd_code', 'key', 'value'])
    res['q1_join'] = dict(q1_rows=int(len(Q)), matched=int(len(m)))
    res['q1_rows_by_eta'] = {int(e): int(n) for e, n in m.eta.value_counts().sort_index().items()}
    res['q1_value_by_eta'] = {int(e): q(g.value) for e, g in m.groupby('eta')}
    zr = m[m.metal == 'Zr']
    res['zr_q1'] = {int(e): dict(value=q(g.value), ratio=q(g.ratio)) for e, g in zr.groupby('eta')}
    nk = m[m.key == "('Zr', 1, 4, 'C_haptic', 'trigonal', 'none')"]
    res['nuwfam_key_rows'] = nk[['csd_code', 'eta', 'value', 'ratio']].round(3).to_dict('records')
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('ratio_max_hist_eta5',)}, indent=1))


if __name__ == '__main__':
    main()
