"""eta2 face-plane package, Step 2: refit the eta2 S_orient entries under eta2_face_plane (geometry only).

  repro    flag off: refit the eta2 keys (k 12 / 112 / 212) from the archived QO observations with the production
           pipeline (fit_orientation on QO + per_sub_rows) and compare with constraints_2026-10-18.json.
  select   flag on: the HAPTIC_FEATS rule for eta2 (drop a feature the other two predict with held-out R^2 >= 0.8,
           boosted trees) on the eta2 face rows; and the eta2 lean range rule (max |asin| of the 2.5 / 97.5 %
           per-substituent H and X bend quantiles) on the faces that use the shared plane.
  fit      flag on: write NEW (DB v5) = production DB with the eta2 keys (k 12 / 112 / 212) replaced. The whole QO
           table is fitted through the production entry point (fit.qo.fit_from_observations -> orientation.fit_rows:
           eta2 rows recomputed under the flag, every face kept) from the QO observations with raw_face_elems
           (diag/eta2_step6_reextract.py: the archived rows, unchanged); every non-eta2 key must come out as in the
           production DB, or nothing is written. Void: v1 (_eta2.json, [tilt, bend]), v2 (_eta2b.json, 2.6 A cut),
           v3 (_eta2c.json, bonded cut with r_cov(C)), v4 (_eta2d.json, bonded cut with per-atom radii). The bonded
           cut went to the far-field / typing package (owner, 2026-11-01).
Run with JACOBIK_PROFILE_OVERRIDES='{"eta2_face_plane": true}' for select / fit. The protected DB is only read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jacobik.stats import orientation as O
from fit import qo as QO

OBS = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/archive/data/refit/observations')
DATA = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/db')
PROD = DATA / 'constraints_2026-10-18.json'
NEW = Path('research/data/jgen/campaign/eta2/constraints_2026-11-01_eta2e.json')  # v5: every eta2 face, no cut
REOBS = Path('research/data/jgen/campaign/eta2/observations')
OUT = Path('research/data/jgen/campaign/eta2')
DS = ('X1', 'X2', 'X3', 'LNf')
ETA2 = (12, 112, 212)


def load():
    df = pd.concat([pd.read_parquet(OBS / f'observations_QO_{d}.parquet') for d in DS], ignore_index=True)
    return df[df.k == 12].reset_index(drop=True)


def recompute_face(df):
    """g0 (tilt) and g1 (face-mean bend) of every eta2 face row from its raw geometry under the active flag."""
    g = []
    for r in df.itertuples():
        F = np.asarray(r.raw_face, float).reshape(-1, 3)
        S = np.asarray(r.raw_subs, float).reshape(-1, 4)
        f = O.haptic_features(np.zeros(3), F, bool(r.raw_closed), [(F[int(s[0])], s[1:]) for s in S])
        g.append(f if f is not None else [np.nan] * 3)
    g = np.asarray(g)
    return df.assign(g0=g[:, 0], g1=g[:, 1], g2=g[:, 2]).dropna(subset=['g0', 'g1', 'g2'])


def qualifies(df):
    out = []
    for r in df.itertuples():
        S = np.asarray(r.raw_subs, float).reshape(-1, 4)
        out.append(len(S) > 0 and min((S[:, 0] == 0).sum(), (S[:, 0] == 1).sum()) >= 2)
    return np.asarray(out)


def same_entry(a, b, tol=1e-9):
    """Same key structure (n, component count, feats, meta) and parameters within tol (refits of the production DB
    agree to <= 2e-14: BLAS / joblib last-bit noise)."""
    if set(a) != set(b) or a['n'] != b['n'] or a.get('feats') != b.get('feats') or a.get('meta') != b.get('meta'):
        return False
    for f in ('weights', 'means', 'covs'):
        x, y = np.asarray(a[f], float), np.asarray(b[f], float)
        if x.shape != y.shape or np.abs(x - y).max() > tol:
            return False
    return True


def fit(df):
    return QO.fit_orientation(QO.fit_rows(df))


def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    prod = json.loads(PROD.read_text())
    if mode == 'repro':
        assert not O.ETA2_FACE_PLANE
        ent = fit(df)
        old = {k: v for k, v in prod.items() if k.startswith('QO:(') and int(k[4:].split(',')[0].rstrip(')')) in ETA2}
        both = [k for k in old if k in ent]
        same_n = [k for k in both if old[k]['n'] == ent[k]['n']]
        same = [k for k in same_n if np.shape(old[k]['means']) == np.shape(ent[k]['means'])
                and np.allclose(old[k]['means'], ent[k]['means'], atol=1e-6)]
        res = dict(prod_keys=len(old), refit_keys=len(ent), shared=len(both), same_n=len(same_n), identical=len(same),
                   n_diff=[(k, old[k]['n'], ent[k]['n']) for k in both if k not in same_n][:8],
                   missing=sorted(set(old) - set(ent))[:10], extra=sorted(set(ent) - set(old))[:10])
        print(res)
        (OUT / 'step2_repro.json').write_text(json.dumps(res, indent=1) + '\n')
        return
    assert O.ETA2_FACE_PLANE, 'run with eta2_face_plane on'
    dq = recompute_face(df)
    if mode == 'select':
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        G = dq[['g0', 'g1', 'g2']].to_numpy(float)
        r2 = {}
        for i, name in enumerate(O.HAPTIC_FEATURES):
            Xo = np.delete(G, i, axis=1)
            a, b, ya, yb = train_test_split(Xo, G[:, i], test_size=0.25, random_state=0)
            r2[name] = round(float(HistGradientBoostingRegressor(random_state=0).fit(a, ya).score(b, yb)), 3)
        q = qualifies(dq)
        ps = QO.per_sub_rows(dq[q])
        lean = {}
        for cls, off in O.PER_SUB_OFFSET.items():
            v = ps[ps.k == 12 + off]['g1'].to_numpy(float)
            lo, hi = np.percentile(v, [2.5, 97.5])
            lean[cls] = dict(n=int(len(v)), q025=round(float(lo), 3), q975=round(float(hi), 3),
                             deg=round(float(np.degrees(max(abs(np.arcsin(lo)), abs(np.arcsin(hi))))), 1))
        res = dict(face_rows=int(len(dq)), qualifying_faces=int(q.sum()), r2_from_other_two=r2, lean_range=lean)
        print(json.dumps(res, indent=1))
        (OUT / 'step2_select.json').write_text(json.dumps(res, indent=1) + '\n')
        return
    if mode == 'fit':
        from fit import qo
        full = pd.concat([pd.read_parquet(REOBS / f'observations_QO_{d}.parquet') for d in DS], ignore_index=True)
        ent = qo.fit_from_observations(full, n_jobs=6)
        is_e2 = lambda k: k.startswith('QO:(') and int(k[4:].split(',')[0].rstrip(')')) in ETA2  # noqa: E731
        p_qo = {k for k in prod if k.startswith('QO:') and not is_e2(k)}
        n_qo = {k for k in ent if not is_e2(k)}
        off = sorted(k for k in p_qo & n_qo if not same_entry(prod[k], ent[k]))
        check = dict(non_eta2_prod=len(p_qo), non_eta2_refit=len(n_qo), missing=sorted(p_qo - n_qo)[:10],
                     extra=sorted(n_qo - p_qo)[:10], differ=len(off), differ_first=off[:10])
        print(json.dumps(check, indent=1))
        if check['missing'] or check['extra'] or off:
            (OUT / 'step2_fit_v5_FAILED.json').write_text(json.dumps(check, indent=1) + '\n')
            raise SystemExit('non-eta2 QO keys do not reproduce the production DB: nothing written')
        e2 = {k: v for k, v in ent.items() if is_e2(k)}
        new = {k: v for k, v in prod.items() if not is_e2(k)}
        new.update(e2)
        assert not NEW.exists(), f'{NEW} exists'
        NEW.write_text(json.dumps(new))
        # the runtime keys model choices (q5_model, q1_variant, ...) on the sidecar: without it every parent's Q5
        # changed (all JACOBIK_DB arms before 2026-11-01 evening ran without one)
        pm = json.loads(PROD.with_name(PROD.stem + '.meta.json').read_text())
        pm['eta2_refit'] = dict(base=PROD.name, eta2_keys='k 12/112/212 refit under eta2_face_plane, every eta2 face',
                                script='research/bench/tools/jgen/diag/eta2_step2_refit.py fit')
        NEW.with_name(NEW.stem + '.meta.json').write_text(json.dumps(pm, indent=2) + '\n')
        rows = QO.fit_rows(full)
        res = dict(db=str(NEW), non_eta2_check=check, eta2_face_rows=int((full.k == 12).sum()),
                   fitted_eta2_face_rows=int((rows.k == 12).sum()), eta2_keys=len(e2),
                   eta2_keys_prod=sum(is_e2(k) for k in prod), dropped=sorted(set(k for k in prod if is_e2(k)) - set(e2))[:20],
                   total_keys=len(new), feats=sorted({tuple(e['feats']) for e in e2.values()}))
        print(json.dumps(res, indent=1))
        (OUT / 'step2_fit_v5.json').write_text(json.dumps(res, indent=1) + '\n')

if __name__ == '__main__':
    main(sys.argv[1])
