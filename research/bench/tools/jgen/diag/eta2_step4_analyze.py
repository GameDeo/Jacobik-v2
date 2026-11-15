"""eta2 package, Step 4: evaluate the holdout as research/campaigns/ETA2_CONTRACT.md amendments 2 and 3 say
(arms ctl = old production, pl5 = eta2 plane b1fcfae58190, capgb = pl5 + haptic cap + group bend = production 273f2ccbf892).

eta2 faces come from the crystal of each pool manifest (ref_X, reference order): bonded C-C pairs (<= 1.6 A) with both
M-C <= ETA2_BOND_RATIO_MAX (r_M + r_C), whose connected set of such carbons has size 2; mapped to builder order through
gen_to_ref_idx (builder[g2r[j]] = reference atom j). Per pool row: the smallest geminal H-C-H angle at a face carbon,
and the smallest shared-plane tilt (orientation.eta2_plane_normal; edge-on if < 0.5).
Usage: python eta2_step4_analyze.py [ROOT] (default research/data/jgen/campaign/eta2/holdout)
"""
import json
import sys
from pathlib import Path

import numpy as np

from jacobik.stats import orientation as O
from jacobik.core.chemistry import covalent_radius as _covalent_radius, METALS

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research/data/jgen/campaign/eta2/holdout')
ARMS = tuple(a for a in ('ctl', 'pl5', 'capgb') if (ROOT / a / 'records').exists())
H_MAX = O.PER_SUB_H_MAX


def faces_of(sym, X):
    m = next(i for i, s in enumerate(sym) if s in METALS)
    rs = _covalent_radius(sym[m]) + _covalent_radius('C')
    near = [i for i, s in enumerate(sym) if s == 'C' and np.linalg.norm(X[i] - X[m]) <= O.ETA2_BOND_RATIO_MAX * rs]
    adj = {i: [j for j in near if j != i and np.linalg.norm(X[i] - X[j]) <= 1.6] for i in near}
    seen, out = set(), []
    for i in near:
        if i in seen:
            continue
        comp, stack = set(), [i]
        while stack:
            k = stack.pop()
            if k not in comp:
                comp.add(k)
                stack += adj[k]
        seen |= comp
        if len(comp) == 2:
            out.append(tuple(sorted(comp)))
    return m, out


def subs_of(sym, X, a, face, m):
    out = []
    for j, s in enumerate(sym):
        if j in face or j == m:
            continue
        if np.linalg.norm(X[j] - X[a]) <= 1.2 * (_covalent_radius(s) + _covalent_radius(sym[a])):
            out.append(j)
    return out


def row_measures(sym, X, m, faces):
    gem, tilt = 180.0, None
    for f in faces:
        F = X[list(f)] - X[m]
        S = []
        for k, a in enumerate(f):
            subs = subs_of(sym, X, a, f, m)
            S += [(F[k], X[j] - X[m]) for j in subs]
            hs = [j for j in subs if sym[j] == 'H' and np.linalg.norm(X[j] - X[a]) < H_MAX]
            for p in range(len(hs)):
                for q in range(p + 1, len(hs)):
                    u, v = X[hs[p]] - X[a], X[hs[q]] - X[a]
                    gem = min(gem, float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))))
        n = O.eta2_plane_normal(F, S)
        if n is not None:
            c = F.mean(0)
            t = abs(float(n @ (-c / np.linalg.norm(c))))
            tilt = t if tilt is None else min(tilt, t)
    return gem, tilt


def read_xyz(p):
    lines = p.read_text().splitlines()
    return np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + int(lines[0])]])


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(d.mean()), 2), round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]


def main():
    draw = json.load(open(ROOT / 'ETA2_HOLDOUT.json'))['draw']
    stratum = {c: s for s, cs in draw.items() for c in cs}
    per = {}
    for arm in ARMS:
        for rec in sorted((ROOT / arm / 'records').glob('*.json')):
            r = json.loads(rec.read_text())
            code, seed = r['code'], int(r['seed'])
            e = dict(status=r['status'], top1=r.get('top1_dE'), floor=r.get('floor_dE'), n_rows=r.get('n_rows'),
                     replay_fallbacks=r.get('replay_fallbacks'),
                     gem_min=None, n_gem30=0, n_edge=0, n_faces=None)
            pm = r.get('pool_manifest')
            if r['status'] == 'ok' and pm and Path(pm).exists():
                P = json.load(open(pm))
                sym, RX, g2r = P['ref_symbols'], np.asarray(P['ref_X'], float), P['gen_to_ref_idx']
                m_ref, faces_ref = faces_of(sym, RX)
                bsym = [None] * len(sym)
                for j, g in enumerate(g2r):
                    bsym[g] = sym[j]
                m, faces = g2r[m_ref], [tuple(g2r[a] for a in f) for f in faces_ref]
                e['n_faces'] = len(faces)
                gems = []
                for row in sorted(Path(pm).parent.glob(f'{code}_{arm}_s{seed}_row*.xyz')):
                    g, t = row_measures(bsym, read_xyz(row), m, faces)
                    gems.append(g)
                    e['n_gem30'] += g < 30
                    e['n_edge'] += t is not None and t < 0.5
                e['gem_min'] = round(min(gems), 1) if gems else None
            per.setdefault(code, {}).setdefault(arm, {})[seed] = e
    out = {'per': per, 'gates': {}}
    draw['D'] = json.load(open(ROOT / 'ETA2_HOLDOUT_D.json'))['draw']['D']
    elig = {c for c, v in json.load(open(ROOT / 'cap_eligible.json')).items() if v[0] > 0}
    out['replay_fallbacks'] = {a: int(sum((per[c][a][s].get('replay_fallbacks') or 0) for c in per if a in per[c]
                                          for s in per[c][a])) for a in ARMS}

    def ok_seeds(c, arms):
        return [s for s in (11, 12, 13) if all(per.get(c, {}).get(a, {}).get(s, {}).get('status') == 'ok' for a in arms)]

    def paired(codes, a, b):
        dt, df, coll = [], [], []
        for c in codes:
            ok = ok_seeds(c, (a, b))
            if not ok:
                continue
            dt.append(np.mean([per[c][a][s]['top1'] - per[c][b][s]['top1'] for s in ok]))
            df.append(np.mean([per[c][a][s]['floor'] - per[c][b][s]['floor'] for s in ok]))
            coll += [max(abs(per[c][a][s]['top1'] - per[c][b][s]['top1']), abs(per[c][a][s]['floor'] - per[c][b][s]['floor']))
                     for s in ok]
        return dict(parents=len(dt), d_top1=boot(dt), d_floor=boot(df), max_abs_diff=round(max(coll), 4) if coll else None)

    def geo(codes, a):
        rs = [per[c][a][s] for c in codes for s in per.get(c, {}).get(a, {})]
        return dict(gem30=int(sum(r['n_gem30'] for r in rs)), edge_on=int(sum(r['n_edge'] for r in rs)),
                    gem_min=min((r['gem_min'] for r in rs if r['gem_min'] is not None), default=None))

    def fail_one(a, b):
        return [(c, s) for c in per for s in (11, 12, 13)
                if (per[c].get(b, {}).get(s, {}).get('status') == 'ok') != (per[c].get(a, {}).get(s, {}).get('status') == 'ok')]

    G = out['gates']
    held = draw['A'] + draw['B'] + draw['C']
    E = sorted({c for c in draw['A'] + draw['B'] if c in elig} | set(draw['D']))
    NE = [c for c in held if c not in elig]
    ABD = draw['A'] + draw['B'] + draw['D']
    # amendment 2: pl5 vs ctl
    g2 = {st: dict(pl5_vs_ctl=paired(draw[st], 'pl5', 'ctl'), geo={a: geo(draw[st], a) for a in ARMS}) for st in 'ABCD'}
    g2['G1_pass'] = g2['A']['geo']['pl5']['gem30'] == 0 and g2['A']['geo']['pl5']['edge_on'] <= g2['A']['geo']['ctl']['edge_on']
    g2['G2_pass'] = all(g2[st]['geo']['pl5']['gem30'] <= g2[st]['geo']['ctl']['gem30'] for st in 'BD')
    g2['G4_failed_pl5_only'] = fail_one('pl5', 'ctl')
    G['amendment2_pl5'] = g2
    # amendment 3: capgb vs pl5
    if 'capgb' in ARMS:
        g3 = dict(geometry_ABD={a: geo(ABD, a) for a in ARMS},
                  E=dict(set=len(E), capgb_vs_pl5=paired(E, 'capgb', 'pl5')),
                  not_eligible=dict(set=len(NE), capgb_vs_pl5=paired(NE, 'capgb', 'pl5')),
                  G4_failed_capgb_only=fail_one('capgb', 'pl5'))
        geo3 = g3['geometry_ABD']
        g3["G7'_pass"] = geo3['capgb']['gem30'] == 0 and geo3['capgb']['edge_on'] <= geo3['pl5']['edge_on']
        de = g3['E']['capgb_vs_pl5']['d_top1']
        dn = g3['not_eligible']['capgb_vs_pl5']['d_top1']
        g3['G5_pass_upper_lt_0'] = bool(de and de[2] < 0)
        g3['noninferior_E_upper_le_10'] = bool(de and de[2] <= 10)
        g3['noninferior_not_eligible_upper_le_10'] = bool(dn and dn[2] <= 10)
        G['amendment3_capgb'] = g3
    (ROOT / 'ETA2_HOLDOUT_RESULT.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'replay_fallbacks': out['replay_fallbacks'], **out['gates']}, indent=1))


if __name__ == '__main__':
    main()
