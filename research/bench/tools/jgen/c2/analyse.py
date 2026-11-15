"""Registered parent-clustered Phase 2 comparisons; no relaxation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .rings import population

SETTINGS = Path('research/campaigns/c2/C2_SETTINGS.json')


def bootstrap(values, settings):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return dict(n=0, mean=None, ci=None)
    rng = np.random.default_rng(settings['bootstrap_seed'])
    samples = rng.choice(v, (settings['bootstrap_resamples'], len(v))).mean(1)
    return dict(n=len(v), mean=float(v.mean()), ci=np.quantile(samples, [.025, .975]).tolist())


def rank(scores, energies):
    s, e = np.asarray(scores, float), np.asarray(energies, float)
    ok = np.isfinite(s) & np.isfinite(e)
    if ok.sum() < 2 or np.ptp(s[ok]) == 0 or np.ptp(e[ok]) == 0: return None
    return float(spearmanr(s[ok], e[ok]).statistic)


def unit(record):
    if record.get('status') != 'ok' or 'c2_metrics' not in record or 'c2_timing' not in record:
        return dict(failed=True, hit1=0., hit05=0., shell03=0., shell05=0., distinct=0.)
    coverage = record['coverage']
    rmsd = min((x for x in coverage['rmsd_A'] if x is not None), default=float('inf'))
    shell = record['c2_metrics']['minimum_shell_rmsd_A']
    return dict(failed=False, floor=record['floor_dE'], top1=record['top1_dE'], min_rmsd=rmsd,
        hit1=float(rmsd<=1.), hit05=float(rmsd<=.5), shell03=float(shell<=.3), shell05=float(shell<=.5),
        min_shell=shell, distinct=record['n_rows'], rho=rank(record['scores'], record['energies_kj']),
        wall=record['c2_timing']['end_to_end_s'])


def control_union(records):
    """Metrics for union of three control seeds; union distinct count precomputed."""
    good = [r for r in records if not unit(r)['failed']]
    if not good: return unit(records[0])
    candidates = [unit(r) for r in good]
    best = max(good, key=lambda r: r['top1_score'])
    base = unit(best)
    base['failed'] = len(good) != len(records)
    for key in ['floor', 'min_rmsd', 'min_shell']:
        base[key] = min(c[key] for c in candidates)
    for key in ['hit1', 'hit05', 'shell03', 'shell05']:
        base[key] = max(c[key] for c in candidates)
    base['distinct'] = records[0]['c2_metrics'].get('control_union_distinct')
    base['failed'] |= base['distinct'] is None
    base['rho'] = rank([x for r in good for x in r['scores']], [x for r in good for x in r['energies_kj']])
    base['wall'] = sum(unit(r)['wall'] for r in good)
    return base


def guard(top, control, settings):
    top, control = np.asarray(top, float), np.asarray(control, float)
    if top.ndim != 2 or not len(top):
        return dict(trips=None,mirror=None,rotation_counts=[],null=None,passed=False,status='no_valid_pairs')
    th, n = settings['regression_threshold_kj'], settings['regression_min_seeds']
    trips = int(((top-control>th).sum(1)>=n).sum())
    mirror = int(((top-control < -th).sum(1)>=n).sum())
    nulls = [int(((top-np.roll(top, i, axis=1)>th).sum(1)>=n).sum()) for i in [1, 2]]
    null = max(nulls)
    return dict(trips=trips, mirror=mirror, rotation_counts=nulls, null=null,
                passed=not (trips > null+settings['regression_null_margin'] and trips>=mirror))


def compare(root, arm, settings):
    codes, _ = population(); primary = [c for c in codes if c not in settings['primary_exclusions']]
    values = {}; failures = []; arm_top=[]; c_top=[]; control_seed_top=[]; walls=[]
    def read_record(path):
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as exc:return dict(status='failed',reason=repr(exc))
    for code in codes:
        cr = [read_record(root/'C/records'/f'{code}_{s}.json') for s in settings['seeds']]
        ar = [read_record(root/arm/'records'/f'{code}_{s}.json') for s in settings['seeds']]
        wide=arm=='S1' or ar[0].get('seed_budget')=='single_w48'
        cu = [control_union(cr)]*3 if wide else [unit(r) for r in cr]
        au = [unit(r) for r in ar]
        metrics = {}
        for key in ['floor','top1','min_rmsd','hit1','hit05','min_shell','shell03','shell05','distinct','rho']:
            paired = [(x.get(key), y.get(key)) for x,y in zip(au,cu)]
            eligible = [(x,y) for x,y in paired if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
            if key not in ['hit1','hit05','shell03','shell05','distinct'] and any(x['failed'] for x in au+cu):
                eligible=[]
            metrics[key] = dict(n_seed_pairs=len(eligible), arm=float(np.mean([x for x,y in eligible])) if eligible else None,
                control=float(np.mean([y for x,y in eligible])) if eligible else None,
                delta=float(np.mean([x-y for x,y in eligible])) if eligible else None)
        values[code]=metrics
        if any(x['failed'] for x in au+cu):
            failures.append(code)
        if code in primary and not any(x['failed'] for x in au+cu):
            arm_top.append([x['top1'] for x in au]); c_top.append([x['top1'] for x in cu])
            control_seed_top.append([unit(r)['top1'] for r in cr])
            # Ordinary arm per-seed wall is a per-parent elapsed run divided
            # equally across seeds. S1 repeats the single-seed experiment 3x.
            aw=sum(x['wall'] for x in au)/(3 if wide else 1)
            cw=cu[0]['wall'] if wide else sum(x['wall'] for x in cu)
            walls.append((aw,cw))
    aggregate={}
    for key in next(iter(values.values())):
        aggregate[key]=bootstrap([values[c][key]['delta'] for c in primary if values[c][key]['delta'] is not None],settings)
        for side in ['arm','control']:
            available=[values[c][key][side] for c in primary if values[c][key][side] is not None]
            aggregate[key][side+'_mean']=float(np.mean(available)) if available else None
    guard_result=guard(arm_top,c_top,settings)
    control_null=guard(control_seed_top,control_seed_top,settings)
    guard_result.update(control_null=control_null['null'],control_rotation_counts=control_null['rotation_counts'])
    wall_ratio=None;wall_ci=None
    if walls:
        wall_ratio=float(np.sum(np.asarray(walls)[:,0])/np.sum(np.asarray(walls)[:,1]))
        rng=np.random.default_rng(settings['bootstrap_seed']);w=np.asarray(walls)
        ix=rng.integers(0,len(w),(settings['bootstrap_resamples'],len(w)))
        wr=w[ix,0].sum(1)/w[ix,1].sum(1)
        wall_ci=np.quantile(wr,[.025,.975]).tolist()
    floor=aggregate['floor']['ci'];coverage=aggregate['hit1']['ci'];top1=aggregate['top1']['ci']
    gates=dict(primary=(floor is not None and floor[1]<0) or (coverage is not None and coverage[0]>0),
               top1=top1 is not None and top1[1]<settings['top1_ci_upper_kj'],
               guard=guard_result['passed'],wall=wall_ratio is not None and wall_ratio<=settings['max_wall_ratio'])
    compute_path=root/'compute_checks.json'
    compute=json.loads(compute_path.read_text()) if compute_path.exists() else []
    compute_ok=not any(c['passed'] is False for c in compute if c['arm']==arm)
    return dict(arm=arm,per_parent=values,metrics=aggregate,guard=guard_result,
        wall_ratio=wall_ratio,wall_ci=wall_ci,gates=gates,
        equal_compute_passed=compute_ok,selected=all(gates.values()) and compute_ok,failures=failures)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--arm',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=compare(a.root,a.arm,json.loads(SETTINGS.read_text()))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(result['gates'])


if __name__=='__main__':main()
