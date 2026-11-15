"""Owner-approved CPU regeneration with native DOFs and fixed-key score paths."""
import argparse,hashlib,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tools.jgen import inputs
from tools.jgen.score_repro import CaptureSession,SolveCapture
from tools.jgen.g3c0_distortion import measure
from tools.jgen.bench.run import _set_finish_flags
from tools.jdev import multiseed
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.core.env import MACE_MODEL_PATH
from jacobik.ik.ik_solver import IKSolverConfig,_CLOSURE_W

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3e/native_twist')
ARM_FILE=Path('research/campaigns/g/G3D_ARMS_AMEND4.json')
PROTOCOL=Path('research/campaigns/g/G3E_CPU_TWIST_PROTOCOL.json')


def write_json(path,value):
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def prepare(code,seed,stats):
    sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
    builder,_,err=build_builder_and_inputs(mol)
    if builder is None:raise ValueError(err)
    with xtb.FallbackWatch(raise_on_exit=True):
        ctx,assignments,W,T=multiseed.prepare(builder,seed,device='cpu',angle_bends=True,stats=stats,n_ik_steps=250,
                            ik_config=IKSolverConfig(n_workers=1,cull_at=(),cull_keep=()))
    return builder,ctx,assignments,W,T


def save_capture(path,cap):
    arrays=dict(X_ref=cap.X_full.cpu().numpy(),scores=cap.scores.cpu().numpy(),candidates=cap.candidate_idx_per_b,
                row_groups=cap.row_groups.cpu().numpy())
    for i,t in enumerate(cap.thetas_final):arrays[f'theta_{i}']=t.cpu().numpy();arrays[f'initial_{i}']=cap.theta_inits[i].cpu().numpy()
    tensor_keys=[];other={}
    for k,v in cap.score_kwargs.items():
        if torch.is_tensor(v):arrays[f'kw_{k}']=v.cpu().numpy();tensor_keys.append(k)
        else:other[k]=v
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    return dict(ligands=len(cap.fks),tensor_kwargs=tensor_keys,other_kwargs=other,bend_prior=cap.bend_prior)


def generate(unit,stats):
    code,seed=unit['code'],unit['seed'];directory=ROOT/f'{code}_{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'selection.json').exists():return
    result=dict(unit=unit,pools={});start=time.monotonic()
    arm_map={a['arm']:a for a in json.loads(ARM_FILE.read_text())['arms']}
    for arm in ['C1','R2s']:
        meta_path=directory/f'{arm}.json'
        if meta_path.exists():result['pools'][arm]=json.loads(meta_path.read_text());continue
        _set_finish_flags(arm_map[arm]['flags'])
        builder,ctx,assignments,W,T=prepare(code,seed,stats)
        with xtb.FallbackWatch(raise_on_exit=True),CaptureSession(builder) as session:
            solved=builder._solve_for_assignments(ctx,assignments,W,T)
        session.attach_conformers(solved.conformers)
        assert len(session.captures)==1
        cap=session.captures[0]
        distortion=measure(solved.conformers,ctx,assignments,cap.ge,cap.scorer,builder.geometry)
        states=save_capture(directory/f'{arm}.npz',cap)
        rows=[dict(index=i,key=[c.embed_idx,int(c.assignment_idx),int(c.trial_idx)],score=float(c.score),
                   X_A=[a.position.tolist() for a in c.complex.atoms],distortion=d)
              for i,(c,d) in enumerate(zip(solved.conformers,distortion))]
        meta=dict(arm=arm,unit=unit,rows=rows,assignments=[list(map(int,a)) for a in assignments],state=states,
                  symbols=[a.symbol for a in ctx.mol.atoms],charge=ctx.mol.charge,spin=ctx.mol.spin_state,
                  cpu_layout=dict(n_workers=1,cull_at=[],threads=1,n_steps=250))
        write_json(meta_path,meta);result['pools'][arm]=meta
    r=result['pools']['R2s'];twisted=[v for v in r['rows'] if v['distortion']['starting_geometry']=='octahedral' and v['distortion']['max_q2_deviation_deg']>45]
    if not twisted:result.update(status='failed',reason='no twisted row in regenerated CPU returned pool')
    else:
        selected=max(twisted,key=lambda v:(v['score'],-v['index']));top=max(r['rows'],key=lambda v:v['score'])
        heavy=[i for i,s in enumerate(r['symbols']) if s!='H'];Q=np.asarray(selected['X_A']);partners=[]
        for arm,pool in result['pools'].items():
            for row in pool['rows']:
                d=row['distortion']
                if row['key'][:2]!=selected['key'][:2] or d['max_q2_deviation_deg']>45 or d['class_changed']:continue
                P=np.asarray(row['X_A']);A=P[heavy]-P[heavy].mean(0);B=Q[heavy]-Q[heavy].mean(0)
                u,_,vt=np.linalg.svd(A.T@B);D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);R=u@D@vt
                distance=float(np.sqrt(np.mean(np.sum((A@R-B)**2,axis=1))));partners.append((distance,arm,row['index']))
        if not partners:result.update(status='failed',reason='no same-assignment untwisted CPU partner')
        else:
            distance,arm,index=min(partners)
            result.update(status='selected',twisted=dict(arm='R2s',index=selected['index'],key=selected['key']),
                          top1=dict(arm='R2s',index=top['index'],key=top['key']),
                          untwisted=dict(arm=arm,index=index,key=result['pools'][arm]['rows'][index]['key'],rmsd_A=distance))
    result.pop('pools');result['wall_s']=time.monotonic()-start
    write_json(directory/'selection.json',result)


def load_capture(directory,arm,stats):
    meta=json.loads((directory/f'{arm}.json').read_text());unit=meta['unit']
    flags=next(a['flags'] for a in json.loads(ARM_FILE.read_text())['arms'] if a['arm']==arm);_set_finish_flags(flags)
    builder,ctx,assignments,W,T=prepare(unit['code'],unit['seed'],stats)
    (fks,dpi,ge,scorer),rows=multiseed.seed_rows(builder,ctx,assignments,W,[unit['seed']])
    assert [list(map(int,a)) for a in assignments]==meta['assignments']
    state=meta['state'];z=np.load(directory/f'{arm}.npz',allow_pickle=False)
    kw=dict(state['other_kwargs']);kw.update({k:torch.tensor(z[f'kw_{k}']) for k in state['tensor_kwargs']})
    cap=SolveCapture(fks,scorer,ge,torch.tensor(z['X_ref']),[torch.tensor(z[f'initial_{i}']) for i in range(state['ligands'])],
                     torch.tensor(z['row_groups']),state['bend_prior'],kw,torch.tensor(z['scores']),
                     [torch.tensor(z[f'theta_{i}']) for i in range(state['ligands'])],z['candidates'],0)
    return cap,meta


def vector(cap,trial):return np.concatenate([v[trial].double().numpy() for v in cap.thetas_final])
def unpack(v,sizes):
    splits=np.cumsum([0]+sizes)
    return [v[splits[i]:splits[i+1]].reshape(1,-1) for i in range(len(sizes))]


def evaluate(cap,v,trial):
    sizes=[t.shape[1] for t in cap.thetas_final];z=torch.tensor(v,dtype=torch.float64,requires_grad=True)
    theta=unpack(z,sizes);total=cap.score(theta,trial)
    grad=torch.autograd.grad(total,z)[0].detach().numpy();ref=cap.X_full.double()
    if ref.ndim==3:ref=ref[trial:trial+1]
    X=ref
    for fk,t in zip(cap.fks,theta):X=fk.apply(t,X)
    kwargs={}
    for k,val in cap.score_kwargs.items():
        if torch.is_tensor(val) and val.ndim and val.shape[0]==len(cap.scores):val=val[trial:trial+1]
        kwargs[k]=val
    kwargs['row_groups']=cap.row_groups[trial:trial+1]
    stat,terms=cap.scorer.score(X,**kwargs)
    seam=X.new_zeros(1)
    for f in cap.fks:
        if f._cl is not None:
            residual,_=f.closure_residual_jacobian(X,f.seam_d0(ref));seam-=_CLOSURE_W*residual.square().sum(-1)
    native={k:float(val.detach()) for k,val in terms.items()}
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6','Q7']:native.setdefault(q,0.)
    native.update(bend_prior=float((total-stat.sum()-seam.sum()).detach()),seam=float(seam.sum().detach()))
    Z=float(cap.scorer._agg_norm());sigma=[]
    for f,n in zip(cap.fks,sizes):sigma.extend([.0585]*3+[.08609]*3+[.2402]*f.n_joints+[.08609]*f.n_angle_bends)
    return dict(score_native=float(total.detach()),score_nats=float(total.detach())*Z,normalization=Z,
                terms_native=native,terms_nats={k:v*Z for k,v in native.items()},
                gradient_norm_native=float(np.linalg.norm(grad)),gradient_norm_nats=float(np.linalg.norm(grad*Z)),
                thermal_gradient_norm_nats=float(np.linalg.norm(grad*Z*np.asarray(sigma))),dofs=len(v)),X[0].detach().numpy()


_MACE=None

def finish_mace(result,meta,dest):
    global _MACE
    if _MACE is None:
        from jacobik.core.energy import MACEPolarEvaluator
        _MACE=MACEPolarEvaluator(model='polar-1-s',device='cpu',dtype='float64',mode='energy-only',threads=1,model_path=str(MACE_MODEL_PATH)).load()
    energies=_MACE.energies([(meta['symbols'],np.asarray(X),meta['charge'],meta['spin']) for X in result['mace_positions']])
    result['mace_energy_kj']=(np.asarray(energies)*96.4853321233).tolist();result['status']='ok'
    write_json(dest,result)


def path_unit(unit,stats):
    directory=ROOT/f"{unit['code']}_{unit['seed']}";dest=directory/'path.json'
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved['status']=='score_path_complete' and 'mace_positions' in saved:
            finish_mace(saved,json.loads((directory/'R2s.json').read_text()),dest)
        return
    selected=json.loads((directory/'selection.json').read_text());result=dict(selection=selected)
    if selected['status']!='selected':result.update(status='failed',reason=selected['reason']);write_json(dest,result);return
    captures={};metas={}
    for arm in {'R2s',selected['untwisted']['arm']}:
        captures[arm],metas[arm]=load_capture(directory,arm,stats)
    observations={};positions={}
    for role in ['twisted','top1','untwisted']:
        identity=selected[role];cap=captures[identity['arm']];row=metas[identity['arm']]['rows'][identity['index']];trial=row['key'][2]
        v=vector(cap,trial);value,X=evaluate(cap,v,trial)
        residual=float(np.sqrt(np.mean(np.sum((X-np.asarray(row['X_A']))**2,axis=1))));error=value['score_native']-row['score']
        observations[role]=dict(value,score_error=error,reconstruction_rmsd_A=residual,key=row['key'])
        if residual>1e-5 or abs(error)>1e-4:raise ValueError(f'native-state verification failed: {role} rmsd {residual}, score error {error}')
        positions[role]=X
    write_json(directory/'gradients.json',dict(status='ok',observations=observations))
    cap=captures['R2s'];tw=selected['twisted'];partner=selected['untwisted'];trial=tw['key'][2]
    v0=vector(cap,trial);other=captures[partner['arm']];v1=vector(other,partner['key'][2]);sizes=[t.shape[1] for t in cap.thetas_final]
    heavy=[i for i,s in enumerate(metas['R2s']['symbols']) if s!='H'];P=positions['untwisted'];Q=positions['twisted']
    u,_,vt=np.linalg.svd((P[heavy]-P[heavy].mean(0)).T@(Q[heavy]-Q[heavy].mean(0)));D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);G=u@D@vt
    ref=other.X_full.double();ref=ref if ref.ndim==2 else ref[partner['key'][2]]
    offset=0
    for fk,n in zip(other.fks,sizes):
        pivot=ref[fk._donor_pos_t].mean(0).numpy() if fk._donor_pos_t.numel() else ref.mean(0).numpy()
        axis=v1[offset+3:offset+6];v1[offset:offset+3]=(pivot+v1[offset:offset+3])@G-pivot
        axis=Rotation.from_matrix(G.T@Rotation.from_rotvec(axis).as_matrix()).as_rotvec()
        length=np.linalg.norm(axis)
        if length>1e-10:axis=min([axis+2*np.pi*k*axis/length for k in [-1,0,1]],key=lambda x:np.linalg.norm(x-v0[offset+3:offset+6]))
        v1[offset+3:offset+6]=axis
        sl=slice(offset+6,offset+6+fk.n_joints);v1[sl]=v0[sl]+np.arctan2(np.sin(v1[sl]-v0[sl]),np.cos(v1[sl]-v0[sl]));offset+=n
    endpoint,X=evaluate(cap,v1,trial);rmsd=float(np.sqrt(np.mean(np.sum((X-P@G)**2,axis=1))))
    if rmsd>1e-5:raise ValueError(f'common-reference aligned endpoint mismatch {rmsd}')
    values=[];xyz=[]
    for t in np.linspace(0,1,21):
        value,X=evaluate(cap,v0+(v1-v0)*t,trial);value['t']=float(t);values.append(value);xyz.append(X)
    minimum=min(range(21),key=lambda i:values[i]['score_nats'])
    result.update(status='score_path_complete',native_observations=observations,path=values,path_minimum_index=minimum,
                  barrier_nats=values[0]['score_nats']-values[minimum]['score_nats'],interior_minimum=0<minimum<20,
                  endpoint_score_change_nats=values[-1]['score_nats']-values[0]['score_nats'],
                  barrier_terms_nats={k:values[0]['terms_nats'][k]-values[minimum]['terms_nats'][k] for k in values[0]['terms_nats']},
                  common_reference_rmsd_A=rmsd,mace_positions=[xyz[i].tolist() for i in [0,20,minimum]])
    write_json(dest,result)
    finish_mace(result,metas['R2s'],dest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['generate','paths'],required=True);parser.add_argument('--indices',required=True);args=parser.parse_args()
    torch.set_num_threads(1);xtb.require();ROOT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(PROTOCOL.read_text());stats=stats_from_db(inputs.DB_PATH)
    for index in map(int,args.indices.split(',')):
        unit=protocol['units'][index];directory=ROOT/f"{unit['code']}_{unit['seed']}";directory.mkdir(parents=True,exist_ok=True)
        try:(generate if args.stage=='generate' else path_unit)(unit,stats)
        except Exception as exc:
            dest=directory/('selection.json' if args.stage=='generate' else 'path.json')
            saved=json.loads(dest.read_text()) if dest.exists() else {}
            saved.update(status='mace_failed' if saved.get('status')=='score_path_complete' else 'failed',unit=unit,reason=str(exc),traceback=traceback.format_exc())
            write_json(dest,saved)
        print(args.stage,unit,json.loads((directory/('selection.json' if args.stage=='generate' else 'path.json')).read_text())['status'],flush=True)

if __name__=='__main__':main()
