"""CPU-only recovery/score-path diagnostic for the frozen exported R2s rows."""
import argparse,hashlib,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tools.jgen import inputs
from tools.jgen.score_repro import _cast_scorer_tensors
from tools.jdev import multiseed
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.core.env import MACE_MODEL_PATH
from jacobik.ik.kernels import dihedral
from jacobik.ik import angle_bends as AB
from jacobik.ik.ik_solver import _BEND_PER_Z,_CLOSURE_W

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B')
SOURCE=ROOT/'g3d/clean24'
OUT=ROOT/'g3e/twist'


def align(P,Q,heavy,metal):
    a=P[heavy];b=Q[heavy];ac=a.mean(0);bc=b.mean(0)
    u,_,vt=np.linalg.svd((a-ac).T@(b-bc));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);R=u@d@vt
    Y=(P-ac)@R+bc
    return Y-Y[metal]  # native FK/scorer frame has its metal at the origin


class Context:
    def __init__(self,code,seed,stats):
        shard,dataset=inputs.frame_shards()[code];self.mol=inputs.load_mol(code,shard,dataset)
        builder,_,err=build_builder_and_inputs(self.mol)
        if builder is None:raise ValueError(err)
        with xtb.FallbackWatch(raise_on_exit=True):
            self.ctx,self.assignments,W,T=multiseed.prepare(builder,seed,device='cpu',angle_bends=True,stats=stats)
            (self.fks,self.dpi,self.ge,self.scorer),rows=multiseed.seed_rows(builder,self.ctx,self.assignments,W,[seed])
        self.initial,self.refs,self.ideal,self.candidates=rows[0]
        self.symbols=[a.symbol for a in self.ctx.mol.atoms]
        self.sizes=[6+f.n_joints+f.n_angle_bends for f in self.fks]
        self.splits=np.cumsum([0]+self.sizes)
        self.metal=self.ge.metal_pos_idx
        self.heavy=[i for i,s in enumerate(self.symbols) if s!='H']
        _cast_scorer_tensors(self.scorer,torch.float64,torch.device('cpu'))

    def ref(self,trial):
        return (self.refs if self.refs.ndim==2 else self.refs[trial]).double().detach()

    def unpack(self,v):
        return [v[self.splits[i]:self.splits[i+1]].reshape(1,-1) for i in range(len(self.sizes))]

    def xyz(self,theta,ref):
        X=ref
        for f,t in zip(self.fks,theta):X=f.apply(t,X)
        return X

    def total(self,theta,ref,assignment):
        X=self.xyz(theta,ref)
        stat,terms=self.scorer.score(X,row_groups=torch.tensor([assignment]))
        bend=stat.new_zeros(1)
        cfg=self.ctx.bend_prior
        if cfg:
            for f,t in zip(self.fks,theta):
                if f.n_angle_bends:
                    free=getattr(f,'_bend_free',None)
                    free=torch.clamp(free,min=float(cfg['free_deg'])) if free is not None else cfg['free_deg']
                    bend+=AB.energy_prior(t[:,6+f.n_joints:],f._bend_k,free,cfg['scale'])
            if _BEND_PER_Z:bend/=float(self.scorer._agg_norm())
        seam=stat.new_zeros(1)
        for f in self.fks:
            if f._cl is not None:
                residual,_=f.closure_residual_jacobian(X,f.seam_d0(ref));seam-=_CLOSURE_W*residual.square().sum(-1)
        return (stat+bend+seam).sum(),dict(terms,bend_prior=bend,seam=seam),X

    def initialization(self,ref,target,mode,trial):
        result=[]
        for li,(f,n) in enumerate(zip(self.fks,self.sizes)):
            t=np.zeros(n)
            if mode=='native':t[6:]=self.initial[li][trial,6:].double().numpy()
            elif mode=='dihedral' and f.n_joints:
                for j,(a,b,c,d) in enumerate(zip(f._flanking_i,f._pivot_j,f._pivot_k,f._flanking_l)):
                    p=torch.as_tensor(target,dtype=torch.float64)
                    delta=float(dihedral(p[a][None],p[b][None],p[c][None],p[d][None])-dihedral(ref[a][None],ref[b][None],ref[c][None],ref[d][None]))
                    t[6+j]=math.atan2(math.sin(delta),math.cos(delta))
            internal=f.apply(torch.tensor(t)[None],ref)[0].detach().numpy()
            idx=f._lig_pos_t.cpu().numpy();P=internal[idx];Q=target[idx];pc=P.mean(0);qc=Q.mean(0)
            u,_,vt=np.linalg.svd((P-pc).T@(Q-qc));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);R=u@d@vt
            pivot=ref[f._donor_pos_t].mean(0).numpy() if f._donor_pos_t.numel() else ref.mean(0).numpy()
            t[:3]=qc-pc@R+pivot@R-pivot;t[3:6]=Rotation.from_matrix(R.T).as_rotvec()
            result.append(t)
        return np.concatenate(result)

    def fit(self,target,trial):
        ref=self.ref(trial);target_t=torch.tensor(target,dtype=torch.float64)[None]
        attempts=[];best=None
        for mode in ['zero','dihedral','native']:
            init=self.initialization(ref,target,mode,trial)
            def objective(v):
                z=torch.tensor(v,dtype=torch.float64,requires_grad=True);X=self.xyz(self.unpack(z),ref)
                loss=(X-target_t).square().mean();g=torch.autograd.grad(loss,z)[0]
                return float(loss.detach()),g.detach().numpy()
            res=minimize(objective,init,jac=True,method='L-BFGS-B',options=dict(maxiter=1500,ftol=1e-16,gtol=1e-11,maxls=50,maxcor=40))
            theta=self.unpack(torch.tensor(res.x,dtype=torch.float64))
            v=[]
            for t in theta:
                a=t.numpy().ravel().copy();a[3:6]=Rotation.from_rotvec(a[3:6]).as_rotvec();a[6:]=np.arctan2(np.sin(a[6:]),np.cos(a[6:]));v.extend(a)
            v=np.asarray(v);X=self.xyz(self.unpack(torch.tensor(v)),ref)[0].detach().numpy()
            rmsd=float(np.sqrt(np.mean(np.sum((X-target)**2,axis=1))))
            attempts.append(dict(mode=mode,rmsd_A=rmsd,iterations=int(res.nit),success=bool(res.success),message=str(res.message)))
            if best is None or rmsd<best[0]:best=(rmsd,v,X)
            if rmsd<=1e-5:break
        return best,attempts,ref

    def evaluate(self,v,ref,assignment):
        z=torch.tensor(v,dtype=torch.float64,requires_grad=True)
        score,terms,X=self.total(self.unpack(z),ref,assignment)
        g=torch.autograd.grad(score,z)[0].detach().numpy();sigma=[]
        for f,n in zip(self.fks,self.sizes):
            sigma.extend([.0585]*3+[.08609]*3+[.2402]*f.n_joints+[.08609]*f.n_angle_bends)
        return dict(total=float(score.detach()),terms={k:float(x.sum().detach()) for k,x in terms.items()},
                    gradient_norm=float(np.linalg.norm(g)),thermal_gradient_norm=float(np.linalg.norm(g*np.asarray(sigma))),
                    predicted_gain_kT=float(.5*np.sum((g*np.asarray(sigma))**2)),dofs=len(v)),X[0].detach().numpy()


def nearest_angles(a,b,sizes):
    b=b.copy();offset=0
    for n in sizes:
        axis=b[offset+3:offset+6];length=np.linalg.norm(axis)
        if length>1e-10:
            options=[axis+2*np.pi*k*axis/length for k in [-1,0,1]]
            b[offset+3:offset+6]=min(options,key=lambda x:np.linalg.norm(x-a[offset+3:offset+6]))
        sl=slice(offset+6,offset+n);b[sl]=a[sl]+np.arctan2(np.sin(b[sl]-a[sl]),np.cos(b[sl]-a[sl]));offset+=n
    return b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=10);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);xtb.require()
    manifest=json.loads(Path('research/campaigns/g/G3E_TWIST_ROWS.json').read_text())
    path=OUT/'result.json'
    result=json.loads(path.read_text()) if path.exists() else dict(rows={},manifest_sha256=hashlib.sha256(Path('research/campaigns/g/G3E_TWIST_ROWS.json').read_bytes()).hexdigest())
    stats=stats_from_db(inputs.DB_PATH);contexts={};started=time.monotonic()
    for index,item in enumerate(manifest['rows'][:args.limit]):
        key=f"{item['code']}_{item['seed']}"
        if key in result['rows']:continue
        row=dict(identity=item);recovered={}
        try:
            ctx=contexts.setdefault(key,Context(item['code'],item['seed'],stats))
            recovered={}
            twisted_record=json.loads((SOURCE/'R2s/records'/f'{key}.json').read_text())
            Xtw=np.asarray(twisted_record['distortion'][item['twisted']['index']]['X_A'],float)
            for role in ['twisted','top1','untwisted']:
                src=item[role];r=json.loads((SOURCE/src['arm']/'records'/f'{key}.json').read_text());d=r['distortion'][src['index']]
                assert d['symbols']==ctx.symbols
                target=np.asarray(d['X_A'],float)
                if role!='twisted':target=align(target,Xtw,ctx.heavy,ctx.metal)
                original_assignment=d['starting_assignment']
                assignment=next(i for i,a in enumerate(ctx.assignments) if list(a)==original_assignment)
                trial=src['key'][2];(residual,v,X),attempts,ref=ctx.fit(target,trial)
                score,X=ctx.evaluate(v,ref,assignment)
                recovered[role]=dict(residual_A=residual,fit_attempts=attempts,evaluation=score,score_stored=src['score'],score_error=score['total']-src['score'],
                                     theta=v.tolist(),reference_trial=trial,starting_assignment_index=assignment)
                if residual>1e-5 or abs(score['total']-src['score'])>1e-4:
                    raise ValueError(f"{role} recovery failed: RMSD={residual:.6g}, score error={score['total']-src['score']:.6g}")
            row['recovery']=recovered
            first=np.array(recovered['twisted']['theta']);last=nearest_angles(first,np.array(recovered['untwisted']['theta']),ctx.sizes)
            # Use one common reference for both endpoint coordinates. Verify it explicitly.
            ref=ctx.ref(recovered['twisted']['reference_trial']);assignment=recovered['twisted']['starting_assignment_index']
            endpoint,Xend=ctx.evaluate(last,ref,assignment)
            if abs(endpoint['total']-recovered['untwisted']['evaluation']['total'])>1e-4:
                raise ValueError('endpoint references differ; a common FK path cannot reproduce the untwisted endpoint')
            values=[];positions=[]
            for t in np.linspace(0,1,21):
                value,X=ctx.evaluate(first+(last-first)*t,ref,assignment);value['t']=float(t);values.append(value);positions.append(X.tolist())
            minimum=min(range(21),key=lambda i:values[i]['total'])
            row.update(status='score_path_complete',recovery=recovered,path=values,path_minimum_index=minimum,
                       barrier_nats=values[0]['total']-values[minimum]['total'],interior_minimum=0<minimum<20,
                       endpoint_score_change=values[-1]['total']-values[0]['total'],
                       barrier_terms={k:values[0]['terms'][k]-values[minimum]['terms'][k] for k in values[0]['terms']},
                       mace_positions=[positions[0],positions[20],positions[minimum]],symbols=ctx.symbols,
                       charge=ctx.mol.charge,spin=ctx.mol.spin_state)
        except Exception as exc:
            row.update(status='failed',reason=str(exc),traceback=traceback.format_exc())
            if 'recovered' in locals():row['recovery']=recovered
        result['rows'][key]=row;result['wall_s']=time.monotonic()-started
        path.write_text(json.dumps(result,indent=2,allow_nan=False));print(key,row['status'],row.get('reason',''),flush=True)
    pending=[(k,v) for k,v in result['rows'].items() if v['status']=='score_path_complete']
    if pending:
        from jacobik.core.energy import MACEPolarEvaluator
        mace=MACEPolarEvaluator(model='polar-1-s',device='cpu',dtype='float64',mode='energy-only',threads=1,model_path=str(MACE_MODEL_PATH)).load()
        for key,row in pending:
            try:
                energies=mace.energies([(row['symbols'],np.asarray(X),row['charge'],row['spin']) for X in row['mace_positions']])
                row['mace_energy_kj']=(np.asarray(energies)*96.4853321233).tolist();row['status']='ok'
            except Exception as exc:row['status']='mace_failed';row['reason']=str(exc)
            result['wall_s']=time.monotonic()-started;path.write_text(json.dumps(result,indent=2,allow_nan=False));print(key,row['status'],flush=True)

if __name__=='__main__':main()
