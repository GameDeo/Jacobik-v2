"""Once-only stage analysis with registered statistics; gate thresholds have no CLI overrides."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.jgen.c2.analyse import bootstrap, unit
from .comparison import compare

SETTINGS = Path('research/campaigns/c2/C2B_SETTINGS.json')
NAMED = 'MIDQEY FEKKAJ DAMVAQ USAPOU HEDJOP NUCKUS VOBGUR QEZVEY NUWFAM JEGPUJ VAVBIE AZUSIA CBTYCO aqua_La3 FELSAU CAXTAX SIVBUV'.split()


def upper(metrics, key, bound):
    ci = metrics[key]['ci']
    return ci is not None and ci[1] < bound


def energy_gate(metrics):
    return upper(metrics, 'top1', 2) and (upper(metrics, 'floor', 0) or upper(metrics, 'top1', 0))


def curvature_valid(summary):
    return (summary.get('complete') is True and summary.get('scope') == 'Q1-Q7'
            and summary.get('q7_scorer_in_scope') is True
            and bool(summary.get('fk_prior_settings'))
            and all(isinstance(summary.get(k), (int, float)) and np.isfinite(summary[k])
                    for k in ('median_ratio', 'rank_agreement')))


def in_curvature_band(summary):
    return 1.15 <= summary['median_ratio'] <= 1.40 and summary['rank_agreement'] >= .80


def curvature_rule(control):
    if not curvature_valid(control):return 'missing_q7_control'
    return 'absolute_band' if in_curvature_band(control) else 'control_relative'


def curvature_gate(curvature, control=None):
    control = control or {}
    if not curvature_valid(curvature) or not curvature_valid(control):return False
    if curvature['fk_prior_settings'] != control['fk_prior_settings']:return False
    if in_curvature_band(control):return in_curvature_band(curvature)
    return (control['median_ratio']-.10 <= curvature['median_ratio'] <= control['median_ratio']+.10
            and curvature['rank_agreement'] >= control['rank_agreement']-.03)


def holdout_gate(metrics, arm_edge, control_edge):
    return upper(metrics, 'top1', 2) and arm_edge is not None and control_edge is not None and arm_edge <= control_edge


def wall_gate(ratio):
    return ratio is not None and ratio <= 1.3


def own_null_gate(guard, null):
    return null["trips"] is not None and guard["trips"] is not None and not (
        guard["trips"] > null["trips"]+2 and guard["trips"] >= guard["mirror"])


def pick(results, contrasts=None):
    eligible = [n for n in ('D', 'DS') if n in results and results[n]['qualified']]
    if eligible:return min(eligible, key=lambda n: (results[n]['metrics']['top1']['mean'], n != 'D'))
    if p_fallback(results, contrasts or {})['eligible']:return 'S'
    return None


def p_fallback(results, contrasts):
    checks=dict(d_and_ds_present=all(n in results for n in ('D','DS','S')),
                neither_d_nor_ds_qualifies=not any(results.get(n,{}).get('qualified',False) for n in ('D','DS')),
                interaction_noninferior=False,p_top1_noninferior=False)
    interaction_metrics=contrasts.get('interaction',{})
    if 'top1' in interaction_metrics:
        checks['interaction_noninferior']=upper(interaction_metrics,'top1',2.)
    if 'S' in results:
        checks['p_top1_noninferior']=upper(results['S']['metrics'],'top1',2.)
    return dict(eligible=all(checks.values()),checks=checks,
                basis='C-2b route D correctness qualification; D3 top-1 interaction and S-vs-C upper CIs < +2')


def interaction(comparisons, settings):
    # The same parent is resampled as a cluster across all four arms.
    d, ds, s = (comparisons[n]['per_parent'] for n in ('D', 'DS', 'S'))
    out = {}
    for key in ('floor', 'top1'):
        vals=[]
        for code in d:
            if code in settings['primary_exclusions']:
                continue
            x=[m[code][key]['delta'] for m in (d, ds, s)]
            if all(v is not None for v in x):
                vals.append(x[1]-x[0]-x[2])
        out[key]=bootstrap(vals, settings)
    return out


def read(root, name, code, seed):
    try:
        return json.loads((root/name/'records'/f'{code}_{seed}.json').read_text())
    except (OSError, ValueError):
        return {'status':'failed'}


def extra_counts(root, name, reference, codes, seeds):
    """Catastrophic counts, edge-on counts and failure symmetry for one arm against its reference.

    Catastrophic counts (top1_dE > 200) and edge-on rows (amendment, auditor 2026-11-12; audit_D3_job1 Phase 1b item 5) are summed over the (parent, seed) units
    that are ok in BOTH arms, so a unit that failed in both arms cannot hide the comparison; those units are listed in
    ``edge_on.excluded``. A unit that failed in one arm only is an asymmetric failure (the failures gate). An ok record
    without an edge-on count is unknown, never zero: the counts are then None and the holdout gate fails.

    Failures gate (owner 2026-11-12): the arm fails only on ``arm_failures``, units that failed in the arm while the reference
    was ok. ``control_failures`` (reference failed, arm ok, e.g. EPUPEL's control timeout) are reported, not held against the
    arm. ``asymmetric_failures`` = both lists, kept for reports."""
    catastrophes=[0,0]; edges=[0,0]; known=True; asymmetric=[]; excluded=[]; arm_only=[]; control_only=[]
    for code in codes:
        for seed in seeds:
            rs=[read(root,n,code,seed) for n in (name,reference)]
            ok=[r.get('status')=='ok' for r in rs]
            if ok[0]!=ok[1]:
                asymmetric.append([code,seed])
                (control_only if ok[0] else arm_only).append([code,seed])
            if not all(ok):
                excluded.append([code,seed]);continue
            for i,r in enumerate(rs):
                catastrophes[i]+=int(r['top1_dE']>200)
                e=r.get('d3_geometry',{}).get('edge_on_rows')
                if e is None:
                    known=False
                else:
                    edges[i]+=e
    return dict(catastrophic=dict(arm=catastrophes[0],control=catastrophes[1],passed=catastrophes[0]<=catastrophes[1]),
                edge_on=dict(arm=edges[0] if known else None,control=edges[1] if known else None,excluded=excluded),
                asymmetric_failures=asymmetric,arm_failures=arm_only,control_failures=control_only)


def analyse(root, stage, manifest, reproduction=False):
    settings=json.loads(SETTINGS.read_text());settings['seeds']=manifest['seeds']
    codes=manifest['codes']; names=manifest['arms']; results={}; contrasts={}
    for name in names:
        if name in ('C','Cprime'):continue
        r=compare(root,name,'D',settings,codes=codes,proof=True)
        r['c2b_gates']=r.pop('gates');r['c2b_qualified']=r.pop('qualified')
        extra=extra_counts(root,name,'C',[c for c in codes if c not in settings['primary_exclusions']],settings['seeds'])
        r.update(extra)
        curvature=manifest.get('curvature',{}).get(name,{})
        g=dict(top1=upper(r['metrics'],'top1',2),guard=r['guard']['without_artifacts']['passed'],
               catastrophic=extra['catastrophic']['passed'],wall=wall_gate(r['wall_ratio']),
               failures=not extra['arm_failures'])
        if stage=='stage1':
            g.update(energy=energy_gate(r['metrics']),curvature=curvature_gate(curvature,manifest.get('curvature',{}).get('C',{})))
        if stage=='stage2':
            g={'holdout':holdout_gate(r['metrics'],**dict(arm_edge=extra['edge_on']['arm'],control_edge=extra['edge_on']['control'])),
               'failures':not extra['arm_failures']}   # amendment 2026-11-12: arm-only failures, all stages
        if stage=='stage3':
            null=compare(root,'Cprime','D',settings,codes=codes,proof=True)['guard']['without_artifacts']
            guard=r['guard']['without_artifacts']
            g={'top1':upper(r['metrics'],'top1',2),
               'guard':own_null_gate(guard,null),
               'failures':not extra['arm_failures']}
            r['own_control_null']=null
        r['gates']=g;r['qualified']=all(g.values()) and manifest.get('label')!='DRYRUN' and not reproduction
        results[name]=r
    if all(n in results for n in ('D','DS','S')):
        contrasts['interaction']=interaction(results,settings)
        ds_s=compare(root,'DS','D',settings,codes=codes,proof=True,reference='S')
        extra=extra_counts(root,'DS','S',[c for c in codes if c not in settings['primary_exclusions']],settings['seeds'])
        ds_s['d3_gates']=dict(energy=energy_gate(ds_s['metrics']),guard=ds_s['guard']['without_artifacts']['passed'],
            catastrophic=extra['catastrophic']['passed'],wall=wall_gate(ds_s['wall_ratio']),
            failures=not extra['arm_failures'])
        contrasts['DS_vs_S']=ds_s
        results['DS']['gates']['DS_vs_S']=all(ds_s['d3_gates'].values())
        results['DS']['qualified'] &= results['DS']['gates']['DS_vs_S']
    fallback=p_fallback(results,contrasts)
    decision_allowed=manifest.get('label')!='DRYRUN' and not reproduction
    fallback['qualified']=fallback['eligible'] and decision_allowed and stage=='stage1'
    candidate=pick(results,contrasts) if decision_allowed else None
    if stage=='stage3':candidate=next((n for n in names if n not in ('C','Cprime') and results[n]['qualified']),None)
    named={c:{n:{str(s):{k:read(root,n,c,s).get(k) for k in ('status','top1_dE','floor_dE')} for s in settings['seeds']}
              for n in names} for c in NAMED if c in codes}
    return dict(stage=stage,label=manifest.get('label','D3'),reproduction=reproduction,settings=settings,
                identities=manifest.get('identities',{}),curvature=manifest.get('curvature',{}),
                curvature_rule=curvature_rule(manifest.get('curvature',{}).get('C',{})),p_fallback=fallback,
                n_parents=len(codes),primary_n=sum(c not in settings['primary_exclusions'] for c in codes),
                comparisons=results,contrasts=contrasts,candidate=candidate,candidate_label='P' if candidate=='S' else candidate,
                p_included=candidate in ('S','DS'),named_cases=named,
                missing_named_cases=[c for c in NAMED if c not in codes])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage1','stage2','stage3'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reproduce-c2b',action='store_true')
    a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text())
    out=a.root/'analysis.json'
    if out.exists():raise FileExistsError('stage analysis already exists; no rerun')
    result=analyse(a.root,a.stage,manifest,a.reproduce_c2b)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('stage','label','candidate','n_parents')}))

if __name__=='__main__':main()
