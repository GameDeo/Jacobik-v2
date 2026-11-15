"""Frozen H-3b paired endpoints: zero-bound cal CI, tail report, all-arm guards."""
import argparse
import csv
import json
from pathlib import Path
from tools.jgen.analyze_h3 import cluster_bootstrap, summarize_panel
from tools.jgen.bench.panels import load_manifest,select


def candidate_keys(obj):
    if isinstance(obj,dict):return {k.replace('combined','candidate'):candidate_keys(v) for k,v in obj.items()}
    if isinstance(obj,list):return [candidate_keys(v) for v in obj]
    return obj


def analyze(root,manifest,arms):
    results={}
    for arm in arms:
        panels=[]
        for name in ['cal','tail']:
            codes=sorted({r['code'] for r in select(manifest,name)['units']})
            p=candidate_keys(summarize_panel(root,name,{'NEPNEF'},codes,candidate_arm=arm))
            boot=p['primary_top1_delta']
            p['candidate_arm']=arm
            for code in ['QUTWOT','EPUWAQ','FEXWUC']:
                values=[r['paired_top1_delta_kj'] for r in p['paired_rows'] if r['code']==code]
                p['named_parents'][code]={'seed_differences_kj':values or None,
                                         'mean_difference_kj':sum(values)/len(values) if values else None,
                                         'improved':bool(len(values)==3 and sum(values)<0)}
            if name=='cal':
                sensitivity={}
                for row in p['paired_rows']:
                    if row['code']!='FEKKAJ':sensitivity.setdefault(row['code'],[]).append(row['paired_top1_delta_kj'])
                p['sensitivity_ex_FEKKAJ']=cluster_bootstrap(sensitivity) if p['status']=='ok' else None
            p['cal_ci_upper_below_zero']=bool(boot and boot['ci95_percentile'][1]<0) if name=='cal' else None
            p.pop('primary_gate_pass')
            p['primary_gate_required']=name=='cal' and arm=='combined'
            p['parent_guard_pass']=not p['guard_parents_gt25_two_of_three']
            p['obscene_gt200']['nonincrease']=p['obscene_gt200'].pop('gate_pass')
            p['wall_solve_s']['within_1_10']=p['wall_solve_s'].pop('gate_pass')
            panels.append(p)
        results[arm]={'panels':panels,'valid_pairs_and_guards_pass':all(p['status']=='ok' and p['parent_guard_pass'] for p in panels),
                      'cal_benefit_ci_below_zero':panels[0]['cal_ci_upper_below_zero']}
    primary=bool('combined' in results and results['combined']['valid_pairs_and_guards_pass'] and results['combined']['cal_benefit_ci_below_zero'])
    individual=all(r['valid_pairs_and_guards_pass'] and r['cal_benefit_ci_below_zero'] for arm,r in results.items() if arm!='combined')
    return {'contract':'H3B_CONTRACT.md','arms':results,'package_primary_pass':primary,
            'individual_benefit_evidence_pass':individual,'all_goals_pass':primary and individual}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arms',default='combined,cap,donor')
    args=ap.parse_args();manifest=load_manifest(Path('data/refit/frame_v1/manifest.json'))
    result=analyze(args.root,manifest,args.arms.split(','))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.out.with_suffix('.csv').open('w',newline='') as f:
        fields=['arm','panel','code','seed','control_top1_dE_kj','candidate_top1_dE_kj','paired_top1_delta_kj',
                'control_floor_dE_kj','candidate_floor_dE_kj','paired_floor_delta_kj']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,r in result['arms'].items():
            for p in r['panels']:
                writer.writerows(dict(arm=arm,panel=p['panel'],**row) for row in p['paired_rows'])
    summary={k:v for k,v in result.items() if k!='arms'}
    summary['arms']={arm:{'guard_pass':r['valid_pairs_and_guards_pass'],'cal_benefit_ci_below_zero':r['cal_benefit_ci_below_zero'],
                         'panels':[{k:v for k,v in p.items() if k in ['panel','status','primary_top1_delta','floor_delta','guard_parents_gt25_two_of_three']}
                                   for p in r['panels']]} for arm,r in result['arms'].items()}
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
