"""Exploratory unrelaxed R2/C0 energy comparison from saved builder-order exports."""
import argparse
import contextlib
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
from tools.runkit import mace,resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--structures',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',choices=('cpu','cuda'),required=True);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--profile-only',action='store_true')
    ap.add_argument('--candidate',default='R2',choices=('R2','R2_minus','P2','P2prime','P2a','P2p'))
    ap.add_argument('--control',default='C0',choices=('C0','C1','R2'))
    ap.add_argument('--reuse-energies',help='Reuse control single points only after matching source digest, device and model')
    args=ap.parse_args();control,candidate=args.control,args.candidate
    def digest(parent,arm):
        return hashlib.sha256(json.dumps([parent[arm]['reference'],parent[arm]['conformers']],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reuse=json.loads(Path(args.reuse_energies).read_text()) if args.reuse_energies else None
    src=json.loads(Path(args.structures).read_text());out=Path(args.out)
    result={'source':args.structures,'device':args.device,'threads':args.threads,'parents':{},'failures':{},'arms':{'control':control,'candidate':candidate}}
    guard=resources.gpu_lock('CodexB',timeout=900) if args.device=='cuda' else contextlib.nullcontext()
    with guard:
        ev=mace.Evaluator(args.device,dtype='float64',threads=args.threads)
        # Warm up, then profile a batched pair on the larger of the two pilot parents.
        pilots=list(src['parents'].values())[:2]
        pilot=max(pilots,key=lambda p:len(p[candidate]['conformers'][0]['symbols']))
        ref=pilot[candidate]['reference'];q,spin=ref['charge'],ref['spin']
        structures=[(r['symbols'],r['X_A'],q,spin) for r in pilot[candidate]['conformers'][:2]]
        ev.energies_kj(structures[:1]);t0=time.monotonic();ev.energies_kj(structures)
        elapsed=time.monotonic()-t0;count=sum(1+sum(p[a]['rows'] for a in (control,candidate)) for p in src['parents'].values())
        result['profile']={'structures':2,'atoms_each':len(structures[0][0]),'wall_s':elapsed,'estimated_all_single_point_s':elapsed*count/2,'requested_structures':count}
        print('profile',result['profile'],flush=True)
        if args.profile_only:
            out.write_text(json.dumps(result,indent=2));return
        result['model_sha256']=ev.model_sha256
        for code,parent in src['parents'].items():
            try:
                assert all(parent[a].get('score_reproduction') is not False for a in (control,candidate)),'score reproduction failed'
                reference=parent[control]['reference'];assert reference==parent[candidate]['reference']
                q,spin=reference['charge'],reference['spin']
                structures=[(reference['symbols'],reference['X_A'],q,spin)]
                cached={}
                if reuse is not None:
                    assert reuse['device']==args.device and reuse['model_sha256']==result['model_sha256']
                    old=reuse['parents'][code][control]
                    assert old['source_digest']==digest(parent,control),'cached control geometry differs'
                    cached[control]=old
                for arm in (control,candidate):
                    if arm in cached:continue
                    for row in parent[arm]['conformers']:
                        assert row['symbols_order']=='builder/conf.complex.atoms'
                        assert len(row['symbols'])==len(row['X_A'])
                        assert np.isfinite(np.asarray(row['X_A'])).all(),'nonfinite exported coordinates'
                        structures.append((row['symbols'],row['X_A'],q,spin))
                t0=time.monotonic();E=ev.energies_kj(structures);rec={'reference_energy_kj':float(E[0]),'energy_wall_s':time.monotonic()-t0}
                assert np.isfinite(E).all(),'nonfinite MACE energy'
                if cached:
                    assert abs(float(E[0])-reuse['parents'][code]['reference_energy_kj'])<1e-5,'cached reference energy differs'
                    rec['reference_energy_kj']=reuse['parents'][code]['reference_energy_kj']
                    rec.update(cached)
                offset=1
                for arm in (control,candidate):
                    if arm in cached:continue
                    rows=parent[arm]['conformers'];es=E[offset:offset+len(rows)];offset+=len(rows);de=es-rec['reference_energy_kj']
                    scores=np.array([r['score'] for r in rows]);top=int(scores.argmax());worst=int(de.argmax())
                    rec[arm]={'source_digest':digest(parent,arm),'n_rows':len(rows),'top1_score':float(scores[top]),'top1_delta_E_unrel_kj':float(de[top]),
                        'floor_delta_E_unrel_kj':float(de.min()),'obscene_count':int((de>200).sum()),'obscene_rate':float((de>200).mean()),
                        'worst_delta_E_unrel_kj':float(de[worst]),'worst_row_key':rows[worst]['key'],
                        'energies_kj':es.tolist(),'all_delta_E_unrel_kj':de.tolist(),'scores':scores.tolist(),'solve_s':parent[arm]['wall_s']}
                rec['paired_top1_change_kj']=rec[candidate]['top1_delta_E_unrel_kj']-rec[control]['top1_delta_E_unrel_kj']
                rec['paired_floor_change_kj']=rec[candidate]['floor_delta_E_unrel_kj']-rec[control]['floor_delta_E_unrel_kj']
                rec['paired_score_change']=rec[candidate]['top1_score']-rec[control]['top1_score']
                rec['score_inversion']=rec['paired_score_change']>0 and rec['paired_top1_change_kj']>0
                result['parents'][code]=rec
            except Exception:result['failures'][code]=traceback.format_exc()
            out.write_text(json.dumps(result,indent=2));print(code,result['parents'].get(code,{}).get('paired_top1_change_kj'),result['failures'].get(code,''),flush=True)
    paired=list(result['parents'].values());rng=np.random.default_rng(11)
    summary={'paired_parents':len(paired),'complete':len(paired)==8 and not result['failures'],'bootstrap':'parent clustered percentile 95%, 10000 resamples, seed11; exploratory 8-parent check'}
    if not paired:
        result['summary']=summary;out.write_text(json.dumps(result,indent=2));return
    for key in ('top1','floor'):
        delta=np.array([p[f'paired_{key}_change_kj'] for p in paired]);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        summary[key]={'mean_paired_change_kj':float(delta.mean()),'median_paired_change_kj':float(np.median(delta)),
            'mean_ci95_kj':np.percentile(boot,[2.5,97.5]).tolist(),'parents_improved':int((delta<0).sum()),'parents_worsened':int((delta>0).sum())}
    summary['score_inversion_parents']=[code for code,p in result['parents'].items() if p['score_inversion']]
    for arm in (control,candidate):
        ps=[p[arm] for p in paired];n=sum(p['n_rows'] for p in ps)
        worst=max(result['parents'],key=lambda code:result['parents'][code][arm]['worst_delta_E_unrel_kj'])
        summary[arm]={'median_top1_delta_E_unrel_kj':float(np.median([p['top1_delta_E_unrel_kj'] for p in ps])),
            'mean_top1_delta_E_unrel_kj':float(np.mean([p['top1_delta_E_unrel_kj'] for p in ps])),
            'mean_floor_delta_E_unrel_kj':float(np.mean([p['floor_delta_E_unrel_kj'] for p in ps])),
            'n_rows':n,'obscene_rate':sum(p['obscene_count'] for p in ps)/n,
            'worst_slot':{'parent':worst,'energy_kj':result['parents'][worst][arm]['worst_delta_E_unrel_kj'],'row_key':result['parents'][worst][arm]['worst_row_key']},
            'sum_solve_s':sum(p['solve_s'] for p in ps)}
    summary['solver_time_ratio_candidate_control']=summary[candidate]['sum_solve_s']/summary[control]['sum_solve_s']
    if reuse is not None and candidate == 'R2_minus' and control == 'C0':
        extra={}
        for metric,field in [('top1','top1_delta_E_unrel_kj'),('floor','floor_delta_E_unrel_kj')]:
            values=[]
            for code,p in result['parents'].items():
                old=reuse['parents'][code]['R2']
                assert old['source_digest']==digest(src['parents'][code],'R2'),'plus geometry differs'
                values.append(p[candidate][field]-old[field])
            delta=np.asarray(values);extra_rng=np.random.default_rng(11)
            boot=delta[extra_rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            extra[metric]={'mean_paired_change_kj':float(delta.mean()),'median_paired_change_kj':float(np.median(delta)),
                'mean_ci95_kj':np.percentile(boot,[2.5,97.5]).tolist(),'parents_improved':int((delta<0).sum())}
        summary['minus_against_plus']=extra
    result['summary']=summary;out.write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
