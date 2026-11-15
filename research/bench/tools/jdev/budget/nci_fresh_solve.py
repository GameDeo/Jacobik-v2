"""Owner-requested targeted NCI generation experiment, isolated from production pools."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
CONTROL='--control' in sys.argv
REPAIR='both' if '--repair-both' in sys.argv else ('cp' if '--repair-cp' in sys.argv else ('caps' if '--repair-caps' in sys.argv else None))
ISOMER='--isomer-ref' in sys.argv
CASES=('PEWBOL',) if ISOMER else (('JEFJUZ','XEPKAF') if REPAIR=='cp' else (('JEFJUZ',) if REPAIR in ('caps','both') else ('XEPKAF','PEWBOL','JEFJUZ','FETSUU','BULGAT','CBTYCO','YIYXEJ')))
os.environ.update(JACOBIK_PROFILE='u1',JACOBIK_PROFILE_OVERRIDES=json.dumps({'nci':not CONTROL}),
                  MKL_THREADING_LAYER='SEQUENTIAL',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
from common import stats_from_db, pool_path, pool_energy
from pools import setup,set_cases,rank_terms
from nci_targeted_rescore import CANDIDATE


def run():
    import torch
    from jacobik.core import profile
    from research.archive.code.interactions import nci
    from jacobik.core.energy import MACEPolarEvaluator
    from tools.jdev.cases import get_case
    from tools.jdev.multiseed import solve_seeds
    from tools.jdev.env import fingerprint,tree_hash
    from research.archive.code.refit.nci_refit import digest
    assert nci.NCI_ON== (not CONTROL) and profile.get('oxidation_model')=='training'
    torch.set_num_threads(1)
    out=ROOT/('data/refit/reports/n2_fresh_control_2026-10-18' if CONTROL else 'data/refit/reports/n2_fresh_solve_2026-10-18')
    if REPAIR:
        out=ROOT/f'data/refit/reports/n2_repair_{REPAIR}_{"off" if CONTROL else "on"}_2026-10-18_v2'
    if ISOMER:
        out=ROOT/f'data/refit/reports/n2_pewbol_isoref_{"off" if CONTROL else "on"}_2026-10-18'
    out.mkdir(exist_ok=True)
    model=ROOT/'tools/jdev/MACE-POLAR-1-S.model'
    config=dict(fingerprint=fingerprint(CANDIDATE,model),db_sha256=digest(CANDIDATE),
                profile=profile.snapshot(),seeds=[11,12,13],
                scope='Seven selected diagnostic cases, not a dev80 arm. No defaults changed. Fresh NCI-on generation vs stored OS pools; CUDA solver noise remains.')
    if REPAIR:
        config.update(experimental_embedding=REPAIR,implementation='v2_input_aromaticity',cases=list(CASES),scope='Ligand-only embedding counterfactual; no target coordinates or complex relaxation. Experimental tool override, no production/profile changes.')
        import jacobik.builder.engine as engine
        from nci_internal_experiment import trial_ligand
        original=engine._smiles_to_conformer
        interventions=[]
        def patched(smiles,**kw):
            from rdkit import Chem
            rd,x,has_h=original(smiles,**kw)
            donors=kw.get('donors') or []
            base=Chem.MolFromSmiles(smiles)
            eligible=(REPAIR in ('cp','both') and len(donors)==1 and len(donors[0])>=3 and all(base.GetAtomWithIdx(i).GetIsAromatic() for i in donors[0])) or (REPAIR in ('caps','both') and '[N-][N-]' in smiles)
            if eligible:
                old=x.copy();x,_=trial_ligand(smiles,donors,smiles_to_conformer=original)
                # Verify changed internal geometry, independent of rigid placement.
                before=np.linalg.norm(old[:,None]-old[None],axis=-1)
                after=np.linalg.norm(x[:,None]-x[None],axis=-1)
                change=float(np.max(abs(after-before)));assert change>1e-3,(smiles,'intervention did not change internals')
                interventions.append(dict(smiles=smiles,max_internal_distance_change=change))
                for j,p in enumerate(x):rd.GetConformer().SetAtomPosition(j,p.tolist())
            return rd,x,has_h
        engine._smiles_to_conformer=patched
    if ISOMER:
        config.update(reference_isomer=True,cases=list(CASES),scope='Reference donor/site assignment only (diagnostic). No reference coordinates in embedding or scoring; sulfonate orientations remain free. No production changes.')
    manifest=out/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        config['fingerprint']['time']=old['config']['fingerprint']['time']
        assert old['config']==config,'resume config changed'
    manifest.write_text(json.dumps(dict(config=config,status='running'),indent=2)+'\n')
    stats=stats_from_db(CANDIDATE)
    names={c:(n,s) for s in ('dev32','n48') for c,n in set_cases(s)}
    start=time.perf_counter();rows=[]
    for code in CASES:
        name,split=names[code];case=get_case(name);builder=case.make_builder()
        if ISOMER:
            from isomer import isomer_builder
            make,dev=isomer_builder(case);assert make is not None,dev
            builder=make()
        variants=len(builder._ez_combinations())
        files=[out/f'{code}_s{s}.npz' for s in config['seeds']]
        if not all(f.exists() for f in files):
            if variants==1:
                results,info=solve_seeds(builder,seeds=config['seeds'],stats=stats,device='cuda',angle_bends=True,batched_assembly=True)
            else:
                # Preserve engine E/Z enumeration and per-variant assembly exactly.
                results={s:builder.solve(seed=s,stats=stats,device='cuda',angle_bends=True) for s in config['seeds']}
                info={'path':'serial seeds with native E/Z enumeration','variants':variants}
            print(code,'solve',json.dumps(info,default=str),flush=True)
            if REPAIR:
                assert interventions,(code,'no embedding intervention applied')
                info['embedding_interventions']=interventions.copy()
            for seed,r in results.items():
                X=np.array([[a.position for a in c.complex.atoms] for c in r.conformers])
                asg=np.array([c.assignment_idx or 0 for c in r.conformers])
                sc=r.scorer;dev=sc._ge._q6_a.device
                score,bd=rank_terms(sc,torch.tensor(X,dtype=torch.float32,device=dev),torch.tensor(asg,device=dev))
                stored=np.array([c.score for c in r.conformers])
                with torch.no_grad():
                    direct=sc.ranking_score(torch.tensor(X,dtype=torch.float32,device=dev),row_groups=torch.tensor(asg,device=dev))
                error=float(np.max(np.abs(score.cpu().numpy()-direct.cpu().numpy())))
                assert error<2e-4,(code,seed,error)
                # Solver score additionally retains its bend/closure penalties.
                penalty=stored-score.cpu().numpy()
                assert np.max(penalty)<2e-4,(code,seed,'positive penalty',penalty.max())
                Z=float(sc._agg_norm())
                save=dict(X=X,score=stored,rank=score.cpu().numpy(),asg=asg,Z=Z,charge=case.charge,mult=case.mult,
                          symbols=np.array([a.symbol for a in r.conformers[0].complex.atoms]),ref_X=case.ref_X,
                          ref_symbols=np.array(case.ref_symbols),profile=json.dumps(profile.snapshot()),score_error=error,solver_penalty=penalty,
                          info=json.dumps(info,default=str))
                save.update({f'terms_{k}':v.detach().cpu().numpy()*Z if k!='Q7' else v.detach().cpu().numpy() for k,v in bd.items()})
                np.savez_compressed(out/f'{code}_s{seed}.npz',**save)
            del results,r,sc,builder
            torch.cuda.empty_cache()
    # The evaluator is loaded only after all solver work; one GPU workload at a time.
    ev=MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',model_path=model,threads=1)
    for code in CASES:
        _,split=names[code]
        for seed in config['seeds']:
            f=out/f'{code}_s{seed}.npz'
            with np.load(f) as z:
                ef=f.with_suffix('.mace64.json')
                if not ef.exists():
                    sy=z['symbols'].tolist();q=int(z['charge']);m=int(z['mult'])
                    energies=ev.energies([(sy,X,q,m) for X in z['X']]+[(z['ref_symbols'].tolist(),z['ref_X'],q,m)])
                    ef.write_text(json.dumps(dict(ref_eV=float(energies[-1]),dE=((energies[:-1]-energies[-1])*96.4853321233).tolist()))+'\n')
                e=np.array(json.loads(ef.read_text())['dE']);i=int(z['score'].argmax())
                base=pool_path('etkdg_u1_os',split,code,seed);be=pool_energy(base,'mace64')
                with np.load(base) as b:
                    ib=int(b['score'].argmax());terms={k:dict(base=float(b[k][ib]),fresh=float(z[k][i])) for k in z.files if k.startswith('terms_') and k in b}
                row=dict(case=code,seed=seed,conformers=len(e),base_pick=float(be[ib]),fresh_pick=float(e[i]),
                         change=float(e[i]-be[ib]),base_floor=float(be.min()),fresh_floor=float(e.min()),
                         base_regret=float(be[ib]-be.min()),fresh_regret=float(e[i]-e.min()),terms=terms,
                         nci_nats=float(z['terms_NCI'][i]) if 'terms_NCI' in z else 0.,score_error=float(z['score_error']))
                rows.append(row);print(json.dumps(row),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(config=config,pools=rows,elapsed_s=time.perf_counter()-start),indent=2)+'\n')
    assert tree_hash()==config['fingerprint']['src'],'source changed during generation'
    manifest.write_text(json.dumps(dict(config=config,status='complete',pools=len(rows),elapsed_s=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':setup();run()
