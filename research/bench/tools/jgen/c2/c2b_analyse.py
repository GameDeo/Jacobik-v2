"""C-2b routes A–D. Settings/seed list come from the frozen contract.

Development subsets are explicitly labeled and cannot imply qualification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyse import unit, control_union, bootstrap, guard
from .matched_rho import matched_rho
from .rings import population

ARTIFACTS = ['NUHPEL','MIDQEY','UMUBAF','YUCZUS','NUWFAM','CAFWIQ']


def route_gates(metrics, route, wall_ratio, guard_result, compute_ok=True, proof=False):
    def upper(key, bound):
        ci=metrics[key]['ci'];return ci is not None and ci[1]<bound
    def lower(key, bound):
        ci=metrics[key]['ci'];return ci is not None and ci[0]>bound
    gates=dict(wall=wall_ratio is not None and wall_ratio<=1.3,
        guard=guard_result['passed'],top1=upper('top1',2.),compute=compute_ok)
    if route=='A':gates['gain']=upper('floor',0.) or lower('hit1',0.)
    else:gates['floor']=upper('floor',2.)
    if route=='B':
        g=guard_result
        gates['targeted']=g['mirror'] is not None and g['mirror']>=g['null']+3 and g['mirror']>=2*g['trips']
    if route=='C':
        gates['coverage']=lower('hit1',-.03)
        gates['ensemble']=upper('matched_rho',0.) or lower('distinct',0.)
    if route=='D':gates['correctness_proof']=proof
    return gates


def compare(root, arm, route, settings, codes=None, development=False, proof=False):
    codes=codes or population()[0]
    assert set(codes)<=set(population()[0])
    values={};failures=[];tops={};walls=[]
    def read(name,code,s):
        try:return json.loads((root/name/'records'/f'{code}_{s}.json').read_text())
        except (OSError,ValueError) as exc:return dict(status='failed',reason=repr(exc))
    for code in codes:
        cr=[read('C',code,s) for s in settings['seeds']]
        ar=[read(arm,code,s) for s in settings['seeds']]
        wide=ar[0].get('seed_budget')=='single_w48'
        cu=[control_union(cr)]*len(cr) if wide else [unit(r) for r in cr]
        au=[unit(r) for r in ar];bad=any(x['failed'] for x in cu+au)
        if bad:failures.append(code)
        metrics={}
        for key in ['floor','top1','hit1','hit05','min_rmsd','min_shell','shell03','shell05','distinct','rho']:
            pairs=[(x.get(key),y.get(key)) for x,y in zip(au,cu)]
            valid=[(x,y) for x,y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
            if bad and key not in ['hit1','hit05','shell03','shell05','distinct']:valid=[]
            metrics[key]=dict(delta=float(np.mean([x-y for x,y in valid])) if valid else None,
                arm=float(np.mean([x for x,y in valid])) if valid else None,
                control=float(np.mean([y for x,y in valid])) if valid else None,n_seed_pairs=len(valid))
        matched=[]
        if not bad:
            cs=[v for r in cr for v in r['scores']] if wide else None
            ce=[v for r in cr for v in r['energies_kj']] if wide else None
            for a,c in zip(ar,cr):
                v=matched_rho(a['scores'],a['energies_kj'],cs if wide else c['scores'],ce if wide else c['energies_kj'],
                    draws=settings['rho_draws'],seed=settings['rho_seed'])
                if v is not None:matched.append(v)
        metrics['matched_rho']=dict(delta=float(np.mean(matched)) if matched else None,n_seed_pairs=len(matched))
        values[code]=metrics
        if code not in settings['primary_exclusions'] and not bad:
            tops[code]=([x['top1'] for x in au],[x['top1'] for x in cu],[unit(r)['top1'] for r in cr])
            aw=sum(x['wall'] for x in au)/(len(au) if wide else 1)
            cw=cu[0]['wall'] if wide else sum(x['wall'] for x in cu)
            walls.append((aw,cw))
    primary=[c for c in codes if c not in settings['primary_exclusions']]
    aggregate={k:bootstrap([values[c][k]['delta'] for c in primary if values[c][k]['delta'] is not None],settings)
               for k in next(iter(values.values()))}
    guards={}
    for label,exclude in [('with_artifacts',[]),('without_artifacts',settings['guard_artifacts'])]:
        eligible_codes=[c for c in tops if c not in exclude]
        eligible=[tops[c] for c in eligible_codes]
        g=guard([v[0] for v in eligible],[v[1] for v in eligible],settings)
        cn=guard([v[2] for v in eligible],[v[2] for v in eligible],settings)
        g.update(control_null=cn['null'],control_rotation_counts=cn['rotation_counts'],n_parents=len(eligible))
        threshold=settings['regression_threshold_kj'];minimum=settings['regression_min_seeds']
        g['trip_parents']=[c for c in eligible_codes if np.sum(np.asarray(tops[c][0],float)-np.asarray(tops[c][1],float)>threshold)>=minimum]
        g['mirror_parents']=[c for c in eligible_codes if np.sum(np.asarray(tops[c][0],float)-np.asarray(tops[c][1],float) < -threshold)>=minimum]
        guards[label]=g
    wall_ratio=float(np.array(walls)[:,0].sum()/np.array(walls)[:,1].sum()) if walls else None
    wall_ci=None
    if walls:
        w=np.asarray(walls);rng=np.random.default_rng(settings['bootstrap_seed'])
        ix=rng.integers(0,len(w),(settings['bootstrap_resamples'],len(w)))
        ratios=w[ix,0].sum(1)/w[ix,1].sum(1)
        wall_ci=np.quantile(ratios,[.025,.975]).tolist()
    checks=json.loads((root/'compute_checks.json').read_text()) if (root/'compute_checks.json').exists() else []
    # A missing wide compute ledger is not a pass.
    compute_ok=all(any(x['code']==c and x['arm']==arm and x['passed'] is True for x in checks) for c in tops) if route=='C' else True
    gates=route_gates(aggregate,route,wall_ratio,guards['without_artifacts'],compute_ok,proof)
    return dict(arm=arm,route=route,development=development,qualified=(all(gates.values()) if not development else None),
        metrics=aggregate,guard=guards,wall_ratio=wall_ratio,wall_ci=wall_ci,gates=gates,failures=failures,per_parent=values)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--arm',required=True);ap.add_argument('--route',choices=list('ABCD'),required=True)
    ap.add_argument('--settings',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--only',default='');ap.add_argument('--development',action='store_true')
    ap.add_argument('--correctness-proof',action='store_true');a=ap.parse_args()
    settings=json.loads(a.settings.read_text())
    if a.development:assert set(settings['seeds'])<={11,12,13}
    if a.only and not a.development:ap.error('confirmation requires full population')
    if not a.development:
        frozen_root=Path('research/campaigns/c2')
        frozen=json.loads((frozen_root/'C2B_FINALISTS.json').read_text())
        declared=next((x for x in frozen['arms'] if x['arm']==a.arm),None)
        if frozen['status']!='frozen' or declared is None or declared['route']!=a.route:
            ap.error('confirmation analysis must use a frozen finalist and its declared route')
        if settings!=json.loads((frozen_root/'C2B_SETTINGS.json').read_text()):
            ap.error('confirmation analysis settings differ from the frozen settings')
    result=compare(a.root,a.arm,a.route,settings,a.only.split(',') if a.only else None,a.development,a.correctness_proof)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(result['gates'])


if __name__=='__main__':main()
