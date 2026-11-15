"""C-2b comparison arithmetic, generalized to a registered reference and panel.

Kept in the same operation order for full-precision confirmation reproduction.
"""
import json
import numpy as np
from tools.jgen.c2.analyse import unit, control_union, bootstrap, guard
from tools.jgen.c2.matched_rho import matched_rho
from tools.jgen.c2.rings import population
from tools.jgen.c2.c2b_analyse import route_gates

def compare(root, arm, route, settings, codes=None, development=False, proof=False, reference="C"):
    codes=codes or population()[0]
    values={};failures=[];tops={};walls=[]
    def read(name,code,s):
        try:return json.loads((root/name/'records'/f'{code}_{s}.json').read_text())
        except (OSError,ValueError) as exc:return dict(status='failed',reason=repr(exc))
    for code in codes:
        cr=[read(reference,code,s) for s in settings['seeds']]
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

