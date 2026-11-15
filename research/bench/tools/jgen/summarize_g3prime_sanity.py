"""Batched unrelaxed energies and prespecified G-3a′ sanity accounting."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.runkit import mace,resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--energy',action='store_true');args=ap.parse_args()
    root=Path(args.root);p=root/'g3prime_sanity.json';d=json.loads(p.read_text());arms=('P2','R1','R1_x01','R1_x10')
    if args.energy:
        with resources.gpu_lock('CodexB'):
            ev=mace.Evaluator('cuda',dtype='float64',threads=1)
            for code,parent in d['parents'].items():
                try:
                    structures=[(r['symbols'],X,r['charge'],r['spin']) for r in parent['rows'] for X in [r['X0_A']]+[r['arms'][a]['X_A'] for a in arms]]
                    E=ev.energies_kj(structures)
                    for i,r in enumerate(parent['rows']):
                        for j,a in enumerate(arms,1):r['arms'][a]['delta_delta_E_unrel_kj']=float(E[5*i+j]-E[5*i])
                except Exception as e:parent['energy_failure']=repr(e)
                p.write_text(json.dumps(d,indent=2))
    rows=[r for parent in d['parents'].values() for r in parent['rows']]
    result={'rows':len(rows),'failures':d['failures'],'arms':{}}
    for a in arms:
        ar=[r['arms'][a] for r in rows]
        rec={'n':len(ar),'median_recovery':float(np.median([r['fraction'] for r in ar])),
            'median_score_gain':float(np.median([r['gain'] for r in ar])),
            'minimum_score_gain':float(min(r['gain'] for r in ar)),
            'median_energy_change_kj':float(np.median([r['delta_delta_E_unrel_kj'] for r in ar])) if all('delta_delta_E_unrel_kj' in r for r in ar) else None}
        if a!='P2':
            conv=[r['convergence_iteration'][0] for r in ar]
            hist=np.array([r['delta_history_median_max_fraction_min_fraction'] for r in ar])[:,:,0,:]
            rec.update(convergence_iterations=conv,converged_rows=sum(c>=0 for c in conv),
                median_convergence_iteration_among_converged=float(np.median([c for c in conv if c>=0])) if any(c>=0 for c in conv) else None,
                delta_history={'median_of_row_medians':np.median(hist[:,:,0],axis=0).tolist(),
                               'mean_fraction_at_max':np.mean(hist[:,:,1],axis=0).tolist(),
                               'mean_fraction_at_min':np.mean(hist[:,:,2],axis=0).tolist()})
        result['arms'][a]=rec
    old=json.loads((root/'o1_g2_cap30_unrel_gpu_results.json').read_text())
    old_by_key={(code,tuple(r['key'])):r for code,p in old['parents'].items() for r in p['rows']}
    lb=[r['delta_E_unrel_kj'] for p in old['parents'].values() for r in p['rows']]
    result['lbfgs_median_energy_change_kj']=float(np.median(lb))
    for a in arms:
        r=result['arms'][a];e=r['median_energy_change_kj']
        if e is not None:
            r['remaining_energy_gap_difference_of_medians_kj']=e-result['lbfgs_median_energy_change_kj']
            r['median_paired_remaining_energy_gap_kj']=float(np.median([
                row['arms'][a]['delta_delta_E_unrel_kj']-old_by_key[(code,tuple(row['key']))]['delta_E_unrel_kj']
                for code,parent in d['parents'].items() for row in parent['rows']]))
            r['median_remaining_score_gain_to_lbfgs']=float(np.median([row['lbfgs_gain']-row['arms'][a]['gain'] for row in rows]))
    r=result['arms']['R1'];b=result['arms']['P2']
    result['R1_pass']=bool(len(rows)==48 and not d['failures'] and r['minimum_score_gain']>=0 and (
        r['median_recovery']>=b['median_recovery'] or (r['median_energy_change_kj'] is not None and r['median_energy_change_kj']<b['median_energy_change_kj'])))
    rp=root/'g3prime_r2.json'
    if rp.exists():
        rd=json.loads(rp.read_text());paired=[p for p in rd['parents'].values() if 'C0' in p and 'R2' in p]
        result['R2']={'parents':len(paired),'failures':rd['failures'],
            'median_C0_score':float(np.median([p['C0']['top1_score'] for p in paired])),
            'median_R2_score':float(np.median([p['R2']['top1_score'] for p in paired])),
            'median_paired_score_change':float(np.median([p['R2']['top1_score']-p['C0']['top1_score'] for p in paired]))}
    target=root/'g3prime_sanity_summary.json';target.write_text(json.dumps(result,indent=2));print(json.dumps({**result,'arms':{a:{k:v for k,v in r.items() if k not in ('delta_history','convergence_iterations')} for a,r in result['arms'].items()}},indent=2))

if __name__=='__main__':main()
