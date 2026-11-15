"""Frozen u1 control scorer reconstruction and fail-closed pool parity audit.

No solve, MACE, or stored reference energy access. Unknown solver priors are
reported as missing; score parity is never obtained by adding saved residuals.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 JACOBIK_PROFILE='u1',JACOBIK_PROFILE_OVERRIDES='{"nci":false,"deterministic_embed":true}')
import argparse,gc,re
from pathlib import Path
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_assignment import native_build
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream,BudgetPause
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry,verified_pool_labels,checked_mapping
from research.archive.code.interactions.nci_v2.nci_v2_response import RawScorerBaseline


def reconstruct(ref,stats):
    from jacobik import profile
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    if profile.get('nci') or not profile.get('deterministic_embed'):raise ValueError('frozen profile must be set before jacobik import')
    native=native_build(ref,11);shell=native['shell']
    for lig in shell.ligands:decompose_ligand(shell,lig)
    scorer=Scorer(shell,GeometryExtractor(shell,torch.device('cpu')),stats,torch.device('cpu'),
        assignments=native['assignments'],donor_groups=[grp for _,_,grp in native['groups']],ideal=native['ideal'])
    g=checked_mapping([a.symbol for a in shell.atoms],[a.symbol for a in ref.atoms],native['mapping'].gen_to_ref_idx)
    return scorer,np.argsort(g),native


def safe_pool_evaluation(scorer,ref_to_generated,pool,path,overlay=None):
    """Verify map/label provenance before calling scorer (including parity use)."""
    X,info=mapped_geometry(pool,pool_path=path,overlay_path=overlay)
    keys=['score']+[f'terms_Q{i}' for i in range(1,7) if f'terms_Q{i}' in pool]
    labels=verified_pool_labels(pool,keys,pool_path=path,overlay_path=overlay)
    # A second independent graph isomorphism can permute identical donors but
    # does not transform saved asg. Native builder atom order is deterministic;
    # score saved poses in that original order ONLY after mapping validation.
    if not np.array_equal(pool['symbols'],[a.symbol for a in scorer.mol.atoms]):
        raise ValueError('saved_builder_atom_order_not_reconstructible')
    original=np.asarray(pool['X'])
    asg=np.asarray(pool['asg'],dtype=np.int64)
    from tools.jdev.budget.pools import rank_terms
    values=[];terms={k:[] for k in keys if k!='score'}
    for start in range(0,len(X),64):
        budget();Y=torch.as_tensor(original[start:start+64],dtype=torch.float32)
        total,bd=rank_terms(scorer,Y,torch.as_tensor(asg[start:start+64]))
        values.append(total.detach().numpy())
        for k in terms:
            q=k.removeprefix('terms_');terms[k].append(bd[q].detach().numpy()*float(scorer._agg_norm()) if q in bd else np.zeros(len(Y)))
    return dict(score=np.concatenate(values),**{k:np.concatenate(v) for k,v in terms.items()}),labels,info


def comparison(pred,obs):
    pred=np.asarray(pred,float);obs=np.asarray(obs,float);delta=np.abs(pred-obs)
    # Strict relative error with explicit zero handling.
    rel=np.divide(delta,np.abs(obs),out=np.where(delta==0,0.,np.inf),where=obs!=0)
    return dict(max_abs=float(delta.max(initial=0)),max_relative=float(rel.max(initial=0)) if np.isfinite(rel).all() else None,relative_error_unbounded=bool(np.isinf(rel).any()),
        failures=int((rel>1e-6).sum()),count=int(obs.size))


def mandatory_parity(rows):
    expected={'score',*[f'terms_Q{i}' for i in range(1,7)]}
    return bool(rows) and all(row.get('status')=='compared' and set(row.get('errors',{}))==expected
        and all(e['failures']==0 for e in row['errors'].values()) for row in rows)


def run(spec_path,frame_path,db_path,out):
    from jacobik.stats.stats import load_db,StatsLookup
    from jacobik.ik import scorer as SC
    budget(start=True);spec=load_json(spec_path);frame=load_json(frame_path);parents=frame['pilot_cohort']['parents'];clean=MAIN/'data/refit/clean'
    dirs=dict(pilot=MAIN/'data/refit/nci_v2/pilot_v1/pools',recert=MAIN/'data/refit/nci_v2/recert_run_a/pools')
    paths={panel:[p for p in sorted(d.glob('*.npz')) if p.name.split('_')[0] in {q['code'] for q in parents} and
        (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))] for panel,d in dirs.items()}
    db_path=Path(db_path);inputs=[frame_path,db_path,db_path.with_name(db_path.stem+'.meta.json'),*sum(paths.values(),[])]
    for p in parents:
        base=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [base/'xyz'/f"tmQM_{p['shard']}.xyz",base/'bo'/f"tmQM_{p['shard']}.BO",base/'tmQM_y.csv']
    r=Run('phase2-baseline-parity',spec_path,inputs,dict(profile='u1',nci=False,deterministic_embed=True,energy_calls=0),out)
    if r.resumed:return r.path
    print('loading frozen DB',flush=True);stats=StatsLookup(load_db(db_path));budget();gc.collect()
    rows=[];refs=[dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents]
    for parent,ref,error in reference_stream(refs,clean):
        budget();code=parent['code'];print('baseline parity',code,flush=True)
        if error:rows.append(dict(code=code,status='missing_reference',reason=error));continue
        try:scorer,inv,native=reconstruct(ref,stats)
        except BudgetPause:raise
        except Exception as exc:rows.append(dict(code=code,status='reconstruction_failed',reason=f'{type(exc).__name__}: {exc}'));continue
        for panel,pp in paths.items():
            for path in pp:
                if not path.name.startswith(code+'_'):continue
                with np.load(path,allow_pickle=False) as z:
                    if 'error' in z.files:continue
                    try:
                        predictions,observed,info=safe_pool_evaluation(scorer,inv,z,path)
                        errors={k:comparison(predictions[k],v) for k,v in observed.items()}
                        rows.append(dict(panel=panel,code=code,pool=path.name,status='compared',errors=errors,
                            missing_terms=[f'Q{i}' for i in range(1,7) if f'terms_Q{i}' not in z.files],mapping=info,
                            saved_score_includes_solver_bend_and_closure=True))
                    except BudgetPause:raise
                    except Exception as exc:rows.append(dict(panel=panel,code=code,pool=path.name,status='excluded',reason=f'{type(exc).__name__}: {exc}'))
        del scorer,native;gc.collect()
    all_terms=[e for row in rows for e in row.get('errors',{}).values()]
    passed=mandatory_parity(rows)
    write_json(r.path/'reports/parity.json',dict(rows=rows,mandatory_parity_passed=passed,production_labels_enabled=False,
        max_relative=max((e['max_relative'] for e in all_terms),default=None) if all(e['max_relative'] is not None for e in all_terms) else None,
        explanation='Saved final score also includes solver bend prior and possibly seam penalties; neither theta nor original embed/rest geometry is serialized. No residual fitting or energy-based repair is used.',
        profile=dict(rank_agg=SC.RANK_AGG,score_agg=SC.SCORE_AGG)))
    r.finish(dict(pools=len(rows),mandatory_parity_passed=passed,production_labels_enabled=False),
        ['Reference/displaced B labels disabled until mandatory saved-pose parity and solver-prior reconstruction pass.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--baseline-db',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(run(a.spec,a.frame,a.baseline_db,a.out))
