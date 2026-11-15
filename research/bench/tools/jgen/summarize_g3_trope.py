"""Nine-parent trope filter: descriptive paired summaries, no inferential CIs."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarise(root,arm):
    units=[];parents={};failures={}
    total_obscene=total_rows=control_obscene=control_rows=0
    for p in sorted((root/arm).glob('*.json')):
        code=p.stem;a=json.loads(p.read_text());cpath=root/'C1'/p.name
        if not cpath.exists():
            failures[code]='missing C1';continue
        c=json.loads(cpath.read_text())
        if a['status']!='ok' or c['status']!='ok':
            failures[code]={'arm':a.get('reason'),'C1':c.get('reason')};continue
        pp=[]
        for seed in ('11','12','13'):
            sa=a['per_seed'].get(seed,{});sc=c['per_seed'].get(seed,{})
            if sa.get('status')!='ok' or sc.get('status')!='ok':
                failures[f'{code}/{seed}']='failed/missing seed';continue
            row={'code':code,'seed':int(seed),'d_score':sa['top1_score']-sc['top1_score'],
                 'dE':sa['dE_unrel']-sc['dE_unrel'],'d_floor':min(sa['all_dE_unrel'])-min(sc['all_dE_unrel']),
                 'arm_top1_dE':sa['dE_unrel'],'control_top1_dE':sc['dE_unrel'],
                 'd_obscene_rate':sa['obscene_rate']-sc['obscene_rate'],
                 'worst_slot':max(sa['all_dE_unrel']),'worst_slot_row':int(np.argmax(sa['all_dE_unrel'])),
                 'arm_floor':min(sa['all_dE_unrel']),'control_floor':min(sc['all_dE_unrel'])}
            pp.append(row);units.append(row)
            total_obscene+=sa['obscene_count'];total_rows+=sa['n_conf']
            control_obscene+=sc['obscene_count'];control_rows+=sc['n_conf']
        parents[code]={'n_seeds':len(pp),'median_d_score':float(np.median([r['d_score'] for r in pp])),
                      'mean_d_score':float(np.mean([r['d_score'] for r in pp])),
                      'median_dE':float(np.median([r['dE'] for r in pp])),
                      'seeds_worse_25':sum(r['dE']>25 for r in pp),
                      'solve_wall_ratio':a['solve_s']/c['solve_s'],
                      'arm_wall_s':a['wall_s'],'control_wall_s':c['wall_s'],
                      'arm_solve_s':a['solve_s'],'control_solve_s':c['solve_s']}
    bad=[c for c,p in parents.items() if p['seeds_worse_25']>=2]
    complete=len(parents)==9 and len(units)==27 and not failures
    nmedian=sum(p['median_d_score']>0 for p in parents.values())
    nmean=sum(p['mean_d_score']>0 for p in parents.values())
    median_de=float(np.median([r['dE'] for r in units]))
    total_wall=float(sum(p['arm_wall_s'] for p in parents.values())/sum(p['control_wall_s'] for p in parents.values()))
    wall=float(sum(p['arm_solve_s'] for p in parents.values())/sum(p['control_solve_s'] for p in parents.values()))
    return {'arm':arm,'control':'C1','complete':complete,'n_parents':len(parents),'n_seed_units':len(units),
            'parents':parents,'units':units,'failures':failures,'guard_fail_parents':bad,
            'parents_score_rises_median':nmedian,'parents_score_rises_mean':nmean,
            'median_paired_d_score':float(np.median([r['d_score'] for r in units])),
            'mean_paired_dE':float(np.mean([r['dE'] for r in units])),
            'median_paired_dE':median_de,'mean_paired_d_floor':float(np.mean([r['d_floor'] for r in units])),
            'obscene_rate':total_obscene/total_rows,'control_obscene_rate':control_obscene/control_rows,
            'rows_arm':total_rows,'rows_control':control_rows,
            'mean_floor':float(np.mean([r['arm_floor'] for r in units])),
            'control_mean_floor':float(np.mean([r['control_floor'] for r in units])),
            'median_top1_dE':float(np.median([r['arm_top1_dE'] for r in units])),
            'control_median_top1_dE':float(np.median([r['control_top1_dE'] for r in units])),
            'worst_slot_unit':{k:v for k,v in max(units,key=lambda r:r['worst_slot']).items() if k in ('code','seed','worst_slot_row')},
            'worst_slot':max(r['worst_slot'] for r in units),
            'worst_top1':max(r['arm_top1_dE'] for r in units),
            'wall_ratio':wall,'total_wall_ratio_including_queue_and_energy':total_wall,'median_solve_wall_ratio':float(np.median([p['solve_wall_ratio'] for p in parents.values()])),
            'filter_pass':complete and min(nmedian,nmean)>4 and median_de<=0 and not bad and wall<=1.3,
            'note':'No CI claimed for nine parents. A proven >=2/3 guard failure permits early rejection; no missing parent is imputed. Wall >1.3 requires explicit justification.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root);result={a:summarise(root,a) for a in ('P1','P2')}
    (root/'g3_trope_summary.json').write_text(json.dumps(result,indent=2))
    for a,r in result.items():
        print(a,{k:v for k,v in r.items() if k not in ('units','parents')})

if __name__=='__main__':
    main()
