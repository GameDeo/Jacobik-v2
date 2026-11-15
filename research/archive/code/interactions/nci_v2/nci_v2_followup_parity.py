"""No-solve deterministic embed/scorer replay for development P4 diagnosis."""
from research.archive.code.interactions.nci_v2.nci_v2_baseline import reconstruct,safe_pool_evaluation,comparison
import argparse,copy,gc,hashlib,shutil,os,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,DEV,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget
from research.archive.code.interactions.nci_v2.nci_v2_poolio import verified_pool_labels


def deterministic_setup(ref,stats,embed=None):
    from jacobik.inputs import build_builder_and_inputs
    from jacobik.builder.engine import ComplexBuilder
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.ik.fk import ForwardKinematics
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    builder,_,error=build_builder_and_inputs(ref)
    if error:raise ValueError(error)
    # Exact default solve embedding: seed None means ETKDG seed42, MMFF then GFN2.
    shell,rdmols,full=builder._build_shell(False,False,embed_seed=None,chelate_embed=True,embed=embed)
    ideal,groups,n=builder._collect_donor_groups(shell,full)
    pairs,heavy=builder._detect_chelate_pairs(shell,groups,n)
    sym=builder._symmetric_donor_groups(shell,rdmols,groups)
    ctx=SimpleNamespace(mol=shell,ideal=ideal,all_donor_groups=groups,n_donors=n,heavy_graph=heavy,symmetric_donor_groups=sym)
    base=builder._compute_baseline_assignment(shell,ideal,groups,n,pairs,heavy_graph=heavy)
    assignments=[base] if all(l.sites is not None for l in builder.ligands) else builder._run_symmetry_enumeration(ctx,base,False,embed_aware=True)
    for ligand in shell.ligands:decompose_ligand(shell,ligand)
    ge=GeometryExtractor(shell,torch.device('cpu'))
    scorer=Scorer(shell,ge,stats,torch.device('cpu'),assignments=assignments,donor_groups=[g for _,_,g in groups],ideal=ideal)
    fks=[ForwardKinematics(shell,l,torch.device('cpu'),angle_bends=True) for l in shell.ligands]
    return scorer,dict(shell=shell,assignments=assignments,groups=groups,ideal=ideal,fks=fks)


def tensor_manifest(value,prefix=''):
    result={}
    if isinstance(value,(torch.Tensor,np.ndarray)):
        result[prefix]=dict(shape=list(value.shape),dtype=str(value.dtype))
    elif isinstance(value,dict):
        for k,v in value.items():result.update(tensor_manifest(v,prefix+'.'+str(k)))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):result.update(tensor_manifest(v,prefix+f'[{i}]'))
    return result


def signature(scorer):
    ge=scorer._ge
    names=['_q5_i','_q5_j','_q5_k','_q5_l']
    return dict(q5_keys=[repr(x) for x in ge._q5_keys],q5_quads=torch.stack([getattr(ge,k) for k in names],dim=1).tolist(),
        q5_sym=ge._q5_sym,q5_means=scorer._q5_means.tolist(),q5_kappas=scorer._q5_kappas.tolist(),
        q5_weights=scorer._q5_weights.tolist(),normalizer=scorer._agg_norm())


def run(spec_path,frame_path,db_path,out,codes=('ICEYAQ','KOTJIP')):
    budget(start=True)
    import jacobik.builder.engine as engine
    from jacobik.stats.stats import load_db,StatsLookup
    temp=DEV/'tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp);os.environ['TMPDIR']=str(temp)
    binary=Path(__import__('sys').executable).parent/'xtb'
    if binary.exists():engine.XTB_BIN=str(binary)
    if not shutil.which(engine.XTB_BIN):raise ValueError('deterministic GFN2 replay requires xtb executable; no silent fallback')
    frame=load_json(frame_path)
    if codes is None:codes=tuple(p['code'] for p in frame['pilot_cohort']['parents'])
    parents=[p for p in frame['pilot_cohort']['parents'] if p['code'] in codes]
    clean=MAIN/'data/refit/clean';db=Path(db_path)
    pools={panel:[p for p in sorted((MAIN/'data/refit/nci_v2'/part).glob('*.npz')) if p.name.split('_')[0] in codes and
        (p.name.endswith('_control.npz') if panel=='recert' else __import__('re').fullmatch(r'.+_s[0-9]+\.npz',p.name))]
        for panel,part in [('pilot','pilot_v1/pools'),('recert','recert_run_a/pools')]}
    inputs=[frame_path,db,db.with_name(db.stem+'.meta.json'),*sum(pools.values(),[]),Path(engine.XTB_BIN)]
    for p in parents:
        b=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [b/'xyz'/f"tmQM_{p['shard']}.xyz",b/'bo'/f"tmQM_{p['shard']}.BO",b/'tmQM_y.csv']
    r=Run('followup-deterministic-parity',spec_path,inputs,dict(codes=list(codes),no_solve=True,embed_seed=42,embed='gfn2'),out)
    if r.resumed:return r.path
    stats=StatsLookup(load_db(db));budget();rows=[];states={};calls=[]
    original=engine._gfn2_refine
    def traced(mol,X,donors=None):
        budget();t=time.perf_counter();Y=original(mol,X,donors)
        calls.append(dict(atoms=len(X),wall_s=time.perf_counter()-t,changed=not np.array_equal(X,Y)))
        print('GFN2 embed',calls[-1],flush=True);budget();return Y
    engine._gfn2_refine=traced
    for p,ref,error in reference_stream([dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents],clean):
        if error:raise ValueError(error)
        code=p['code'];print('deterministic replay',code,flush=True)
        before,inv,_=reconstruct(ref,stats);after,state=deterministic_setup(ref,stats)
        states[code]=dict(before=signature(before),after=signature(after),
            X_embed=np.asarray([a.position for a in state['shell'].atoms]).tolist(),
            bonds_after=[[b.i,b.j,b.order] for b in state['shell'].bonds],
            assignments=[x.tolist() for x in state['assignments']],donor_groups=[g for _,_,g in state['groups']],
            fks=[dict(n_joints=f.n_joints,n_angle_bends=f.n_angle_bends,n_macro_joints=f.n_macro_joints,
                tensors=tensor_manifest(f.__dict__,f'fks[{i}]'),
                bend_plan_tensors=tensor_manifest(f._bend_plan.__dict__,f'fks[{i}]._bend_plan') if f._bend_plan is not None else {}) for i,f in enumerate(state['fks'])],
            scorer_tensor_manifest=tensor_manifest(after.__dict__,'scorer'),geometry_tensor_manifest=tensor_manifest(after._ge.__dict__,'ge'))
        for panel,paths in pools.items():
            for path in paths:
                if not path.name.startswith(code+'_'):continue
                with np.load(path,allow_pickle=False) as z:
                    if 'error' in z.files:continue
                    a,observed,_=safe_pool_evaluation(before,inv,z,path);b,_,_=safe_pool_evaluation(after,inv,z,path)
                    record=dict(code=code,panel=panel,pool=path.name,before={k:comparison(a[k],v) for k,v in observed.items()},
                        after={k:comparison(b[k],v) for k,v in observed.items()})
                    if 'rank' in z.files:
                        rank=verified_pool_labels(z,['rank'],pool_path=path)['rank']
                        record['after_vs_stored_scorer_rank']=comparison(b['score'],rank)
                        residual=np.asarray(observed['score'])-rank
                        record['stored_final_minus_rank']=dict(min=float(residual.min()),max=float(residual.max()),nonzero=int(np.count_nonzero(residual)))
                        target=r.path/f'reports/predictions/{panel}/{path.name}';target.parent.mkdir(parents=True,exist_ok=True)
                        np.savez(target,scorer_after=b['score'],stored_final=observed['score'],stored_rank=rank)
                    rows.append(record)
        del before,after,state;gc.collect()
    write_json(r.path/'reports/parity.json',dict(rows=rows,states=states,embed_calls=calls,no_solve=True,no_residual_corrections=True))
    r.finish(dict(pools=len(rows),parents=len(parents),embed_calls=len(calls)),['Scorer-only replay excludes unrecorded solver bend/closure priors.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--baseline-db',required=True);p.add_argument('--out',required=True);p.add_argument('--all-parents',action='store_true');a=p.parse_args()
    print(run(a.spec,a.frame,a.baseline_db,a.out,codes=None if a.all_parents else ('ICEYAQ','KOTJIP')))
