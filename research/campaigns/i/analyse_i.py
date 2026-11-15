"""Campaign I analysis against I_CONTRACT.md. Reads records written by run_i_screen.py.

usage: python analyse_i.py --root RUN_DIR --seeds 11,12,13 --out summary.json
All CIs: parent-clustered bootstrap, 10,000 resamples, seed 11 (C2_SETTINGS.json).
"""
import argparse, json
from pathlib import Path
import numpy as np
from tools.jgen.c2.analyse import unit, bootstrap, guard
from tools.jgen.c2.matched_rho import matched_rho
from tools.jgen.c2.rings import population

SET = json.loads(Path('research/campaigns/c2/C2_SETTINGS.json').read_text())
ARTIFACTS = ['NUHPEL', 'MIDQEY', 'UMUBAF', 'YUCZUS', 'NUWFAM', 'CAFWIQ']
ABSENT = ['DAMVAQ', 'FETTAZ', 'FUTSOD', 'XACMOE', 'ZAZREZ', 'LISVOC']
CAPPED = ['CIFZEZ', 'CIXYUD', 'DUHHIX', 'EDOBUV', 'GOTWET', 'HEDJOP', 'NOYTUS', 'NUCKUS', 'NUHPEL', 'QEZVEY', 'SILROY', 'XAQCAT', 'YOVNEC']
NAMED = list(dict.fromkeys(ABSENT + ['BOCRAN'] + CAPPED))


def rd(root, arm, code, s):
    try:
        return json.loads((root / arm / 'records' / f'{code}_{s}.json').read_text())
    except Exception as e:
        return dict(status='failed', reason=repr(e))


def mean_over(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--seeds', default='11,12,13')
    ap.add_argument('--arms', default='A,AS,ASP')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--control', default='C')
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',')]
    codes, _ = population()
    primary = [c for c in codes if c not in SET['primary_exclusions']]
    out = dict(seeds=seeds, n_primary=len(primary), arms={})
    recs = {arm: {c: [rd(a.root, arm, c, s) for s in seeds] for c in codes} for arm in list(dict.fromkeys([a.control] + a.arms.split(',')))}
    failures = {arm: [c for c in codes if any(r.get('status') != 'ok' for r in recs[arm][c])] for arm in recs}
    out['failures'] = failures

    for arm in a.arms.split(','):
        res = dict()
        per = {}
        for c in primary:
            cu = [unit(r) for r in recs[a.control][c]]
            au = [unit(r) for r in recs[arm][c]]
            if any(x['failed'] for x in cu + au):
                continue
            d = {}
            for k in ['floor', 'top1', 'hit1', 'hit05', 'distinct']:
                d[k] = mean_over([x[k] - y[k] for x, y in zip(au, cu)])
                d[k + '_arm'] = mean_over([x[k] for x in au])
                d[k + '_ctl'] = mean_over([x[k] for x in cu])
            d['wall_arm'] = float(sum(x['wall'] for x in au))
            d['wall_ctl'] = float(sum(x['wall'] for x in cu))
            mr = [matched_rho(x['scores'], x['energies_kj'], y['scores'], y['energies_kj'],
                              draws=20, seed=11) for x, y in zip(recs[arm][c], recs[a.control][c])]
            d['matched_rho'] = mean_over(mr)
            d['rho'] = mean_over([x['rho'] - y['rho'] if x['rho'] is not None and y['rho'] is not None else None
                                  for x, y in zip(au, cu)])
            per[c] = d
        res['n_valid_parents'] = len(per)
        res['metrics'] = {k: bootstrap([per[c][k] for c in per if per[c][k] is not None], SET)
                          for k in ['floor', 'top1', 'hit1', 'hit05', 'distinct', 'rho', 'matched_rho']}
        res['wall_ratio'] = float(sum(per[c]['wall_arm'] for c in per) / sum(per[c]['wall_ctl'] for c in per))
        # guard, with and without artefacts
        for label, excl in [('with_artifacts', []), ('without_artifacts', ARTIFACTS)]:
            cs = [c for c in per if c not in excl]
            top = [[unit(r)['top1'] for r in recs[arm][c]] for c in cs]
            ctl = [[unit(r)['top1'] for r in recs[a.control][c]] for c in cs]
            g = guard(top, ctl, SET)
            cn = guard(ctl, ctl, SET)
            g['control_null'] = cn['null']
            th, mn = SET['regression_threshold_kj'], SET['regression_min_seeds']
            g['trip_parents'] = [c for c, t, k in zip(cs, top, ctl) if np.sum(np.array(t) - np.array(k) > th) >= mn]
            g['mirror_parents'] = [c for c, t, k in zip(cs, top, ctl) if np.sum(np.array(t) - np.array(k) < -th) >= mn]
            res.setdefault('guard', {})[label] = g
        # M2 duplicate fraction (returned rows duplicating a better row), capped vs uncapped
        def dupfrac(arm_, subset):
            nd = nr = 0
            for c in subset:
                for r in recs[arm_][c]:
                    m = r.get('i_metrics', {}).get('m2_dupes')
                    if m:
                        nd += m['n_rows_dup_of_better']; nr += m['n_rows']
            return dict(dup_rows=nd, rows=nr, frac=(nd / nr if nr else None))
        capped = [c for c in primary if any(r.get('i_metrics', {}).get('m2_dupes', {}).get('is_capped') for r in recs[a.control][c])]
        res['dup'] = dict(n_capped=len(capped),
                          control_all=dupfrac(a.control, primary), arm_all=dupfrac(arm, primary),
                          control_capped=dupfrac(a.control, capped), arm_capped=dupfrac(arm, capped),
                          control_uncapped=dupfrac(a.control, [c for c in primary if c not in capped]),
                          arm_uncapped=dupfrac(arm, [c for c in primary if c not in capped]))
        # stereo presence (M3 measure + Scope-B label measure), stereo-bearing parents
        stp = [c for c in codes if recs[a.control][c][0].get('i_metrics', {}).get('m3_stereo', {}).get('n_cen', 0) +
               recs[a.control][c][0].get('i_metrics', {}).get('m3_stereo', {}).get('n_dbl', 0) > 0]
        def pres(arm_, key, c):
            return [bool(r.get('i_metrics', {}).get(key, {}).get('present')) for r in recs[arm_][c]]
        res['stereo'] = dict(
            m3_parents=len(stp),
            m3_present_control=sum(any(pres(a.control, 'm3_stereo', c)) for c in stp),
            m3_present_arm=sum(any(pres(arm, 'm3_stereo', c)) for c in stp),
            m3_all_seed_present_control=sum(all(pres(a.control, 'm3_stereo', c)) for c in stp),
            m3_all_seed_present_arm=sum(all(pres(arm, 'm3_stereo', c)) for c in stp),
            scopeB_present_control=sum(any(pres(a.control, 'scope_b_stereo', c)) for c in codes),
            scopeB_present_arm=sum(any(pres(arm, 'scope_b_stereo', c)) for c in codes),
            named={c: dict(m3_control=pres(a.control, 'm3_stereo', c), m3_arm=pres(arm, 'm3_stereo', c),
                           sb_control=pres(a.control, 'scope_b_stereo', c), sb_arm=pres(arm, 'scope_b_stereo', c),
                           top1_arm=[r.get('top1_dE') for r in recs[arm][c]],
                           top1_ctl=[r.get('top1_dE') for r in recs[a.control][c]]) for c in NAMED if c in recs[arm]},
            absent_set_m3_control=sum(any(pres(a.control, 'm3_stereo', c)) for c in ABSENT if c in recs[a.control]),
            absent_set_m3_arm=sum(any(pres(arm, 'm3_stereo', c)) for c in ABSENT if c in recs[arm]))
        # per-class output audit (ASP contract) and classes per parent
        aud = [r['i_metrics']['out_per_class_audit'] for c in primary for r in recs[arm][c] if r.get('i_metrics')]
        res['classes'] = dict(
            mean_classes=mean_over([x['n_classes'] for x in aud]),
            max_rows_per_class=max([x['max_rows_per_class'] for x in aud], default=None),
            all_within_k=all(x['all_within_k'] for x in aud) if aud else None,
            all_within_window=all(x['all_within_window'] for x in aud) if aud else None,
            mean_rows=mean_over([r['n_rows'] for c in primary for r in recs[arm][c] if r.get('n_rows')]),
            ctl_mean_rows=mean_over([r['n_rows'] for c in primary for r in recs[a.control][c] if r.get('n_rows')]))
        # Additional paired parent-level metrics; each bootstrap resamples parents.
        extra = {}
        for key in ['duplicate_fraction', 'classes', 'rows', 'stereo_present', 'wall_ratio']:
            diffs = []
            for c in per:
                def metric(arm_):
                    vals = []
                    for r in recs[arm_][c]:
                        im = r.get('i_metrics', {})
                        if key == 'duplicate_fraction':
                            m = im.get('m2_dupes', {})
                            v = m.get('n_rows_dup_of_better', 0) / max(1, m.get('n_rows', 0))
                        elif key == 'classes':
                            v = im.get('out_per_class_audit', {}).get('n_classes')
                        elif key == 'rows':
                            v = r.get('n_rows')
                        elif key == 'stereo_present':
                            v = float(bool(im.get('m3_stereo', {}).get('present')))
                        else:
                            v = unit(r)['wall']
                        vals.append(v)
                    return mean_over(vals)
                av, cv = metric(arm), metric(a.control)
                if av is not None and cv is not None:
                    diffs.append(av / cv if key == 'wall_ratio' and cv > 0 else av - cv)
            extra[key] = bootstrap(diffs, SET)
        res['additional_parent_clustered_metrics'] = extra
        # storage
        res['storage_bytes'] = sum(p.stat().st_size for p in (a.root / arm / 'xyz').rglob('*') if p.is_file())
        res['storage_bytes_control'] = sum(p.stat().st_size for p in (a.root / a.control / 'xyz').rglob('*') if p.is_file())
        # exactness: arm A vs C on uncapped parents (bit-identical rows) and capped (<= rows)
        if arm == 'A' and a.control == 'C':
            ident = []; viol = []
            for c in primary:
                for r, q in zip(recs['A'][c], recs[a.control][c]):
                    if r.get('status') != 'ok' or q.get('status') != 'ok':
                        continue
                    if c in capped:
                        if r['n_rows'] > q['n_rows']:
                            viol.append(c)
                    else:
                        ident.append(c) if (r['scores'] == q['scores'] and r['energies_kj'] == q['energies_kj']) else viol.append(c)
            res['exactness'] = dict(uncapped_identical_units=len(ident), violations=sorted(set(viol)))
        out['arms'][arm] = res
    if 'AS' in recs and 'A' in recs:
        out['hard_check'] = [dict(code=c, seed=s, a=x.get('top1_score'), as_score=y.get('top1_score'))
                             for c in codes for s, x, y in zip(seeds, recs['A'][c], recs['AS'][c])
                             if x.get('status') == y.get('status') == 'ok'
                             and x.get('top1_score') is not None and y.get('top1_score') is not None
                             and y['top1_score'] < x['top1_score'] - 1e-6]
    a.out.write_text(json.dumps(out, indent=1, default=str))
    for arm, r in out['arms'].items():
        print(arm, {k: (v['mean'], v['ci']) for k, v in r['metrics'].items()}, 'wall', r['wall_ratio'])
        print('  guard', {k: (g['trips'], g['mirror'], g['null'], g['passed']) for k, g in r['guard'].items()})
        print('  dup', r['dup']['control_all'], r['dup']['arm_all'], 'capped', r['dup']['control_capped'], r['dup']['arm_capped'])
        print('  stereo', {k: v for k, v in r['stereo'].items() if k != 'named'})


if __name__ == '__main__':
    main()
