"""Minimal bend-parameter prior and development motion sensitivity estimates."""
# Profile must be established before jacobik import.
from research.archive.code.interactions.nci_v2.nci_v2_followup_parity import deterministic_setup
import argparse,math,sys,tempfile,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,DEV,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,geometry_catalog,geometry_values
from research.archive.code.interactions.nci_v2.nci_v2_poolio import verified_pool_labels
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream


def prior_energy(beta,k,free_degrees,scale=1.):
    beta=np.asarray(beta,float);k=np.asarray(k,float);free=np.deg2rad(np.asarray(free_degrees,float))
    if beta.shape[-1:]!=k.shape or free.shape!=k.shape or np.any(k<0) or np.any(free<0) or scale<0:raise ValueError('invalid bend state shape/constants')
    if not all(np.isfinite(v).all() for v in (beta,k,free,np.asarray(scale))):raise ValueError('nonfinite bend state')
    return .5*scale*np.sum(k*np.maximum(np.abs(beta)-free,0.)**2,axis=-1)


def prior_slope(beta,beta_dot,k,free_degrees,scale=1.):
    prior_energy(beta,k,free_degrees,scale);beta=np.asarray(beta,float);dot=np.asarray(beta_dot,float)
    if dot.shape!=beta.shape or not np.isfinite(dot).all():raise ValueError('invalid bend tangent')
    return scale*np.sum(np.asarray(k)*np.maximum(np.abs(beta)-np.deg2rad(free_degrees),0.)*np.sign(beta)*dot,axis=-1)


def change_bound(total_center_energy,k,amplitude,scale=1.):
    """Single bend-column bound from a nonnegative total prior-energy upper bound."""
    if total_center_energy<0 or k<0 or scale<0:raise ValueError('invalid bound inputs')
    return math.sqrt(2*scale*k*total_center_energy)*abs(amplitude)+.5*scale*k*amplitude**2


def constants(ref):
    from jacobik.inputs import build_builder_and_inputs
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.ik.fk import ForwardKinematics
    builder,_,error=build_builder_and_inputs(ref)
    if error:raise ValueError(error)
    shell,_,_=builder._build_shell(False,False,embed_seed=None,chelate_embed=True,embed=None)
    result=[]
    for ligand in shell.ligands:
        decompose_ligand(shell,ligand);fk=ForwardKinematics(shell,ligand,torch.device('cpu'),angle_bends=True)
        result.append(dict(k=fk._bend_k.numpy() if fk.n_angle_bends else np.empty(0),
            free=np.maximum(fk._bend_free.numpy(),2.) if fk.n_angle_bends else np.empty(0),K=fk.n_angle_bends,macro=fk.n_macro_joints))
    return result


def run(spec_path,frame_path,motion_dir,out):
    import jacobik.builder.engine as engine
    from jacobik import profile
    if not profile.get('deterministic_embed') or not profile.get('bend_per_z'):raise ValueError('requires frozen u1 deterministic/per-Z prior profile')
    budget(start=True);temp=DEV/'tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp);os.environ['TMPDIR']=str(temp);engine.XTB_BIN=str(Path(sys.executable).parent/'xtb')
    frame=load_json(frame_path);parents=frame['pilot_cohort']['parents'];clean=MAIN/'data/refit/clean';motions=sorted(Path(motion_dir).glob('motions/*/*.json'))
    files={panel:[p for p in sorted((MAIN/'data/refit/nci_v2'/part).glob('*.npz')) if p.name.split('_')[0] in {q['code'] for q in parents} and
        (__import__('re').fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))] for panel,part in [('pilot','pilot_v1/pools'),('recert','recert_run_a/pools')]}
    inputs=[frame_path,*motions,*sum(files.values(),[]),Path(engine.XTB_BIN)]
    for p in parents:
        b=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [b/'xyz'/f"tmQM_{p['shard']}.xyz",b/'bo'/f"tmQM_{p['shard']}.BO",b/'tmQM_y.csv']
    r=Run('followup2-prior-response',spec_path,inputs,dict(no_mace=True,selected_fixed_beta=True,scenarios=['zero_beta','free_zone_boundary','observed_baseline_energy_upper_bound']),out)
    if r.resumed:return r.path
    all_constants={};estimates=[];selected=[]
    for p,mol,error in reference_stream([dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents],clean):
        budget();code=p['code'];print('bend sensitivity',code,flush=True)
        if error:raise ValueError(error)
        cs=constants(mol);all_constants[code]=cs;catalog=geometry_catalog(mol)
        for path in motions:
            if path.parent.name!=code:continue
            data=load_json(path);center=data['center'];E0=None
            if center['center']=='baseline':
                pool=next(q for q in files[center['panel']] if q.name==center['pool'])
                with np.load(pool,allow_pickle=False) as z:
                    lab=verified_pool_labels(z,['score','rank','Z'],pool_path=pool);i=center['pose_index'];s=float(lab['score'][i]);rank=float(lab['rank'][i]);Z=float(lab['Z'])
                    eps=2e-6*max(1.,float(np.abs(lab['score']).max()),float(np.abs(lab['rank']).max()))
                    E0=2.479*Z*(max(rank-s,0.)+eps)
            for m in data['selected']:
                X=np.asarray(m['positions']);angles=geometry_values(X,catalog)['bond_angle'];shift=float(np.max(np.abs(angles-angles[2:3]))) if angles.size else 0.
                selected.append(dict(parent=code,center=center['center'],kind=m['definition']['kind'],scan=m['scan'],motion=m['motion_id'],
                    delta_prior_kj_fixed_beta=0. if m['definition']['kind']!='bend' else None,projected_prior_kj_per_t_fixed_beta=0. if m['definition']['kind']!='bend' else None,
                    max_ligand_bond_angle_change_deg=shift,assumption='native beta held fixed; Cartesian beta reconstruction is not proven'))
            for m in data['diagnostics']:
                d=m.get('definition',{})
                if d.get('kind')!='bend' or not m.get('valid'):continue
                li,j=d['ligand'],d['column']
                if j>=cs[li]['K']:raise ValueError('P3 bend column absent from frozen profile')
                k=float(cs[li]['k'][j]);free=float(cs[li]['free'][j]);delta=math.radians(2.)
                zero_local=float(prior_energy(np.array([delta]),np.array([k]),np.array([free])))
                zero_wide=float(prior_energy(np.array([4*delta]),np.array([k]),np.array([free])))
                estimates.append(dict(parent=code,center=center['center'],panel=center['panel'],ligand=li,column=j,k=k,free_degrees=free,
                    zero_beta_local_endpoint_kj=zero_local,zero_beta_8deg_endpoint_kj=zero_wide,
                    free_boundary_local_endpoint_kj=.5*k*delta**2,
                    observed_total_prior_upper_kj=E0,local_abs_delta_upper_kj=change_bound(E0,k,delta) if E0 is not None else None,
                    wider_abs_delta_upper_kj=change_bound(E0,k,4*delta) if E0 is not None else None))
    write_json(r.path/'reports/sensitivity.json',dict(selected=selected,valid_bend_candidates=estimates,constants=all_constants,
        recommendation='bend_prior_required',
        caveats=['No MACE residual y measured; no claim of an observed fraction of physical response.',
            'Selected paths have no bend columns. Zero estimates condition on fixed native beta, not an identified Cartesian beta(X).',
            'Observed score-rank may include closure; it upper-bounds nonnegative bend energy up to declared numerical band.',
            'Unselected 8deg scans are sensitivity calculations, not selected or independently validated paths.']))
    r.finish(dict(selected=len(selected),valid_bend_candidates=len(estimates),mace_calls=0),['Minimal beta state is missing in original pools; estimates are conditional scenarios/bounds.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--motions',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(run(a.spec,a.frame,a.motions,a.out))
