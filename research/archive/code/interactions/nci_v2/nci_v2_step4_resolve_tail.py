"""GPU NCI-v2 active re-solve for the calibration-only tail panel."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch

def run(frame_path,clean_root,db_path,sidecar_path,model_path,out_path,*,codes=None,max_parents=None,mem_fraction=.45):
    from jacobik.stats.stats import StatsLookup,load_db
    from jacobik.benchmark.energy import MACEPolarEvaluator,EV_TO_KJ_MOL
    from research.archive.code.refit.nci_v2_orchestrator import build_builder_and_inputs
    from research.archive.code.interactions.nci_v2.nci_v2_step4_label_gpu import safe_load_mol,safe_smiles_index
    from research.archive.code.refit.nci_v2_s2b_verify import load_npz
    from research.archive.code.interactions.nci_v2.nci_v2_response import artifact_from_json
    from tools.jdev import multiseed as MS

    if not torch.cuda.is_available():raise RuntimeError('tail active re-solve requires CUDA')
    torch.cuda.set_per_process_memory_fraction(mem_fraction,0)
    frame=json.loads(Path(frame_path).read_text());parents=frame['main_allocation']['splits']['calibration']
    by_code={p['representative_code']:p for p in parents}
    tail=['DAMVAQ','HOXJUZ','IFUROQ','JEGPUJ','NEPNEF','SOBHAS','USAPOU','XEYTUT','YUCZUS','ZELTAP']
    selected=[c for c in sorted(tail) if c in by_code and (codes is None or c in set(codes))]
    db=json.loads(Path(db_path).read_text());mp=Path(db_path).with_name(Path(db_path).stem+'.meta.json')
    stats=StatsLookup(load_db(db,meta=json.loads(mp.read_text()) if mp.exists() else None))
    calibrated=artifact_from_json(json.loads(Path(sidecar_path).read_text()))
    eligibility=calibrated.metadata.get('family_eligible',{})
    for family in calibrated.family_scales:
        if eligibility and not eligibility.get(family,False):calibrated.family_scales[family]=0.
    if calibrated.metadata.get('identified') is False or (eligibility and not any(eligibility.values())):
        raise RuntimeError('all NCI scales are unidentified; Part 5b must be skipped')
    stats.nci_v2_artifact=calibrated
    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    evaluator=MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',model_path=model_path,mode='energy-only',threads=1)
    evaluator.load();torch.cuda.synchronize()
    out=Path(out_path);progress=out.with_suffix('.progress.json')
    rows=json.loads(progress.read_text()).get('rows',[]) if progress.exists() else []
    done={r['code'] for r in rows if r.get('status')=='ok' and r.get('complete_parent')}
    t0=time.perf_counter();completed=0
    for code in selected:
        if code in done:continue
        parent=by_code[code]
        mol=safe_load_mol(code,parent['representative_shard'],parent['dataset'],Path(clean_root),smiles)
        if mol is None:
            rows.append(dict(code=code,status='failed',reason='reference_not_found'));continue
        builder,_,err=build_builder_and_inputs(mol)
        if builder is None:
            rows.append(dict(code=code,status='failed',reason='builder:'+str(err)));continue
        t=time.perf_counter()
        result,info=MS.solve_seeds(builder,seeds=[11,12,13],device='cuda',batched_assembly=True,
                                   angle_bends=True,stats=stats,max_rows=None)
        refX=np.asarray([a.position for a in mol.atoms],dtype=np.float64)
        refE=evaluator.energy([a.symbol for a in mol.atoms],refX,mol.charge,mol.spin_state)*EV_TO_KJ_MOL
        for seed in (11,12,13):
            res=result.get(seed)
            if res is None or not res.conformers:
                rows.append(dict(code=code,seed=seed,status='failed',reason='empty_conformers'));continue
            confs=res.conformers
            X=np.asarray([[a.position for a in c.complex.atoms] for c in confs],dtype=np.float64)
            energies=np.asarray([evaluator.energy([a.symbol for a in confs[0].complex.atoms],x,mol.charge,mol.spin_state)
                                 for x in X],dtype=np.float64)*EV_TO_KJ_MOL
            scores=np.asarray([c.score for c in confs],dtype=np.float64)
            top=int(np.argmax(scores));rows.append(dict(code=code,seed=seed,status='ok',
                top1_dE=float(energies[top]-refE),floor_dE=float(energies.min()-refE),
                n_rows=len(X),n_finite=int(np.isfinite(energies).sum()),top1_index=top,
                score_top1=float(scores[top]),elapsed_s=time.perf_counter()-t,
                energy_reference_kj=float(refE),energy_top1_kj=float(energies[top]),
                energy_floor_kj=float(energies.min())))
        rows.append(dict(code=code,status='ok',complete_parent=True,parent_wall_s=time.perf_counter()-t,
                         seed_solve_s=info.get('s_total'),seeds_per_pass=info.get('seeds_per_pass')))
        done.add(code);completed+=1
        progress.write_text(json.dumps(dict(rows=rows,completed_parents=len(done)),separators=(',',':'))+'\n')
        print(f'completed tail parent {len(done)}/{len(selected)}: {code}; wall={time.perf_counter()-t0:.1f}s',flush=True)
        if max_parents is not None and completed>=max_parents:break
    elapsed=time.perf_counter()-t0
    result=dict(schema='nci_v2_step4_tail_resolve_v1',status='descriptive_in_sample',rows=rows,
        new_parents=completed,completed_parents=len(done),elapsed_s=elapsed,
        gpu=evaluator.describe(),mem_fraction=mem_fraction,
        candidate_db_sha256=hashlib.sha256(Path(db_path).read_bytes()).hexdigest(),
        nci_sidecar_sha256=hashlib.sha256(Path(sidecar_path).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        known_failure_parents_touched=sorted(set(selected)&{'JEGPUJ','NEPNEF','HOXJUZ','DAMVAQ','BEBQEE'}))
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','clean-root','db','sidecar','model','out'):p.add_argument('--'+x,required=True)
    p.add_argument('--codes');p.add_argument('--max-parents',type=int);p.add_argument('--mem-fraction',type=float,default=.45)
    a=p.parse_args();r=run(a.frame,a.clean_root,a.db,a.sidecar,a.model,a.out,codes=a.codes.split(',') if a.codes else None,max_parents=a.max_parents,mem_fraction=a.mem_fraction)
    print(json.dumps({k:r[k] for k in ('new_parents','completed_parents','elapsed_s')}))
