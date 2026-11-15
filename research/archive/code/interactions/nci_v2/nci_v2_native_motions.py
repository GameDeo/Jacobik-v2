"""Native solver DOFs applied at exact Cartesian centers, CPU build only."""
from research.archive.code.interactions.nci_v2.nci_v2_assignment import native_build
import argparse,hashlib,re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities,features_torch,FAMILIES
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_extract import unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2_response import AMPLITUDES,generate_motions
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream,BudgetPause
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_poolio import checked_mapping,mapped_geometry,certified_assignment_mask


def rotate(X,atoms,pivot,axis,angle):
    axis=axis/torch.linalg.vector_norm(axis).clamp_min(1e-12);q=X[atoms]-pivot
    v=pivot+q*torch.cos(angle)+torch.linalg.cross(axis.expand_as(q),q)*torch.sin(angle)+(q*axis).sum(-1,keepdim=True)*axis*(1-torch.cos(angle))
    return X.index_copy(0,torch.as_tensor(atoms,dtype=torch.long),v)


def native_dofs(ref,native,X):
    """No IK or center reconstruction: use the native graph subtree indices.

    Macrocycle seam joints require coupled closure and are explicitly omitted;
    their count is returned, never silently treated as unconstrained torsions.
    """
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.ik.fk import ForwardKinematics
    X=torch.as_tensor(X,dtype=torch.float64);shell=native['shell']
    g=checked_mapping([a.symbol for a in shell.atoms],[a.symbol for a in ref.atoms],native['mapping'].gen_to_ref_idx)
    inv=np.argsort(g);metal=next(i for i,a in enumerate(ref.atoms) if a.idx==ref.metal_idx)
    dofs=[];omitted=[]
    for li,lig in enumerate(shell.ligands):
        decompose_ligand(shell,lig);fk=ForwardKinematics(shell,lig,torch.device('cpu'),angle_bends=True)
        atoms=inv[np.asarray(fk._lig_pos_indices)].tolist()
        for kind in ('translation','rotation'):
            for j in range(3):
                axis=torch.eye(3,dtype=X.dtype)[j]
                def path(t,kind=kind,axis=axis,atoms=atoms):
                    if kind=='rotation':return rotate(X,atoms,X[metal],axis,t*AMPLITUDES[kind])
                    return X.index_copy(0,torch.tensor(atoms),X[atoms]+axis*t*AMPLITUDES[kind])
                dofs.append(dict(id=f'lig{li}:{kind}:{j}',kind=kind,path=path,definition=dict(native=True,ligand=li,kind=kind,atoms=atoms,axis=axis.tolist(),pivot='metal' if kind=='rotation' else None)))
        for j in range(fk.n_joints_base+fk.n_bend_joints):
            atoms=inv[fk._downstream[j].numpy()].tolist();a,b=int(inv[fk._pivot_j[j]]),int(inv[fk._pivot_k[j]])
            def path(t,atoms=atoms,a=a,b=b):return rotate(X,atoms,X[a],X[b]-X[a],t*AMPLITUDES['torsion'])
            dofs.append(dict(id=f'lig{li}:torsion:{j}',kind='torsion',path=path,
                definition=dict(native=True,ligand=li,kind='torsion',joint=j,atoms=atoms,axis_atoms=[a,b],legacy_branch=j>=fk.n_joints_base)))
        for j in range(fk.n_angle_bends):
            def path(t,j=j,plan=fk._bend_plan,K=fk.n_angle_bends):
                beta=torch.zeros((1,K),dtype=X.dtype);beta[0,j]=t*AMPLITUDES['bend']
                return plan.apply(X[inv][None],beta)[0,g]
            dofs.append(dict(id=f'lig{li}:bend:{j}',kind='bend',path=path,
                definition=dict(native=True,ligand=li,kind='bend',column=j,
                                theta_column=6+fk.n_joints+j)))
        if fk.n_macro_joints:omitted.append(dict(ligand=li,macro_joints=fk.n_macro_joints,reason='requires_coupled_seam_closure_not_a_free_subtree'))
    return dofs,omitted


def feature_coverage(records,opps,spec):
    """Geometry coverage only. Not a substitute for calibrated |Delta D|."""
    touched={f:np.zeros(len(o),bool) for f,o in opps.opportunities.items()}
    for r in records:
        if not r.get('selected') or not r['valid']:continue
        ft=features_torch(torch.tensor(r['positions'],dtype=torch.float64),opps,spec).features
        for f,v in ft.items():touched[f]|=(v-v[2:3]).abs().amax((0,2)).numpy()>spec['native_motions']['feature_displacement_tolerance']
    result={}
    for f,oo in opps.opportunities.items():
        # Certified root includes the intraligand flag at index 7.
        intra=np.array([bool(o['z_o'][0][7]) for o in oo],dtype=bool)
        result[f]={}
        for label,mask in [('intra_ligand',intra),('metal_mediated',~intra),('all',np.ones(len(oo),bool))]:
            n=int(mask.sum());k=int((touched[f]&mask).sum());result[f][label]=dict(occurrences=n,traversed_features=k,fraction=k/n if n else None,
                D_traversal='unidentified_no_supported_development_surface')
    return result


def run(spec_path,frame_path,certificate_dir,out):
    budget(start=True);spec=load_json(spec_path);frame=load_json(frame_path);parents=frame['pilot_cohort']['parents']
    clean=MAIN/'data/refit/clean';cert=Path(certificate_dir)
    dirs=dict(pilot=MAIN/'data/refit/nci_v2/pilot_v1/pools',recert=MAIN/'data/refit/nci_v2/recert_run_a/pools')
    pools={panel:[p for p in sorted(d.glob('*.npz')) if p.name.split('_')[0] in {q['code'] for q in parents} and
        (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))] for panel,d in dirs.items()}
    inputs=[frame_path,*sum(pools.values(),[]),cert/'manifest.json']
    inputs+=list(cert.glob('masks/*/*.npz'))
    for p in parents:
        base=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [base/'xyz'/f"tmQM_{p['shard']}.xyz",base/'bo'/f"tmQM_{p['shard']}.BO",base/'tmQM_y.csv']
    r=Run('phase2-native-motions',spec_path,inputs,dict(no_energy=True,baseline_centers='one geometry-hash selected eligible pose per pool'),out)
    if r.resumed:return r.path
    reports=[]
    refs=[dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents]
    for parent,ref,error in reference_stream(refs,clean):
        budget();code=parent['code'];print('native motions',code,flush=True)
        if error:reports.append(dict(code=code,status='missing_reference',reason=error));continue
        try:native=native_build(ref,11)
        except BudgetPause:raise
        except Exception as exc:reports.append(dict(code=code,status='native_build_failed',reason=str(exc)));continue
        Xref=np.asarray([a.position for a in ref.atoms]);opps=unique_opportunities(enumerate_opportunities(ref,spec))
        centers=[dict(label='reference',panel='reference',X=Xref)]
        for panel,paths in pools.items():
            for path in paths:
                if not path.name.startswith(code+'_'):continue
                with np.load(path,allow_pickle=False) as z:
                    if 'error' in z.files:continue
                    X,info=mapped_geometry(z,pool_path=path)
                    mask=certified_assignment_mask(cert/f'masks/{panel}/{path.stem}.npz',path,len(X))
                    health,_,_,_=health_metrics(ref,Xref,X,spec['solver_artifact_controls'],z['coord_intact'] if 'coord_intact' in z.files else None)
                    eligible=np.flatnonzero(mask&health['healthy'])
                    if not len(eligible):reports.append(dict(code=code,panel=panel,pool=path.name,status='no_eligible_baseline_center'));continue
                    pick=min(eligible,key=lambda i:hashlib.sha256(np.asarray(X[i],dtype='<f8').tobytes()).digest())
                    centers.append(dict(label='baseline',panel=panel,X=X[pick],pool=path.name,pose_index=int(pick)))
        def validity(Y):
            h,_,_,_=health_metrics(ref,Xref,Y.detach().numpy(),spec['solver_artifact_controls'])
            return dict(valid=h['healthy'],bad_bond=h['bad_bond'],overlap=h['gross_overlap'],coord_intact=h['coord_intact'])
        for ci,c in enumerate(centers):
            budget();dofs,omitted=native_dofs(ref,native,c['X'])
            records=generate_motions(c['X'],opps,group_id=parent['group_id'],complex_id=code,dofs=dofs,
                center=c['label'],validity=validity,schema=spec,wider_multiplier=spec['native_motions']['wider_multiplier'])
            selected=[m for m in records if m.get('selected')]
            report=dict(code=code,panel=c['panel'],center=c['label'],pool=c.get('pool'),pose_index=c.get('pose_index'),
                candidates=len(dofs),selected=len(selected),omitted=omitted,
                rejections=dict(Counter(reason for m in records for reason in m.get('rejection_reasons',[]))),coverage=feature_coverage(records,opps,spec))
            reports.append(report)
            write_json(r.path/f'motions/{code}/{ci}.json',dict(center=report,selected=selected,
                diagnostics=[{k:v for k,v in m.items() if k not in ('positions','dXdt0','fd_positions','half_fd_positions','validity')} for m in records]))
    write_json(r.path/'reports/motions.json',dict(centers=reports,policy=spec['native_motions'],energy_calls=0))
    r.finish(dict(parents=len(parents),centers=sum('candidates' in p for p in reports),energy_calls=0),
        ['Macrocycle seam joints omitted explicitly; coupled closure not implemented.',
         'Feature traversal is geometric only; calibrated D traversal is unidentified without supported development surfaces.',
         'One geometry-hash selected eligible baseline center per pool; not exhaustive baseline-center coverage.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--certificates',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(run(a.spec,a.frame,a.certificates,a.out))
