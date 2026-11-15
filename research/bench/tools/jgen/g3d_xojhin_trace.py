"""Separate new-base R2s COD folding/closure diagnostic, never a policy cohort."""
import json,os
from pathlib import Path
import torch
from jacobik.core import profile
from jacobik.ik import ik_solver as I
from jacobik.ik.kernels import dihedral
from tools.jgen import inputs
from tools.jgen.g3d_diagnostics import native_irprop_trace
from tools.jdev.budget.common import stats_from_db
from jacobik.core.env import build_builder_and_inputs


def main():
    os.environ.pop('JACOBIK_NUM_WORKERS',None);torch.set_num_threads(1)
    shard,dataset=inputs.frame_shards()['XOJHIN'];mol=inputs.load_mol('XOJHIN',shard,dataset)
    builder,_,err=build_builder_and_inputs(mol)
    if builder is None:raise ValueError(err)
    baseline=profile.snapshot();stats=stats_from_db(inputs.DB_PATH)
    original=builder._run_joint_ik;old_hook=I.JointIKSolver.diagnostic_hook
    context={};calls=0;geometry=[]
    def joint(*args,**kwargs):
        nonlocal calls
        context['variant']=calls;calls+=1
        return original(*args,**kwargs)
    def hook(solver,t,data):
        with torch.no_grad():
            X=data['X'];best=int(data['score_loss'].argmin())
            for li,(fk,theta) in enumerate(zip(solver.fks,data['thetas'])):
                if not fk.n_macro_joints:continue
                start=fk._macro_col0-6;end=start+fk.n_macro_joints
                quads=list(zip(fk._flanking_i[start:end],fk._pivot_j[start:end],fk._pivot_k[start:end],fk._flanking_l[start:end]))
                angles=torch.stack([dihedral(X[:,a],X[:,b],X[:,c],X[:,d]) for a,b,c,d in quads],dim=1)*180/torch.pi
                residual,_=fk.closure_residual_jacobian(X,fk.seam_d0(data['X_ref']))
                geometry.append(dict(variant=context['variant'],step=t,ligand=li,rows=len(theta),quads=quads,
                    macro_theta_radians=theta[best,fk._macro_col0:fk._macro_col0+fk.n_macro_joints].tolist(),
                    live_best_row=best,live_best_dihedrals_deg=angles[best].tolist(),
                    dihedral_quantiles_deg=torch.quantile(angles,torch.tensor([.05,.5,.95]),dim=0).tolist(),
                    closure_max_abs_A=float(residual.abs().max()),live_best_closure_A=residual[best].tolist()))
    builder._run_joint_ik=joint;I.JointIKSolver.diagnostic_hook=staticmethod(hook)
    out=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/B/g3d/a3_XOJHIN.json')
    try:
        trace=native_irprop_trace(builder,stats,11,250,50)
        result=dict(status='ok',baseline=baseline,code='XOJHIN',seed=11,steps=250,
                    geometry=geometry,optimizer=trace,scope='separate R2s COD search diagnostic; excluded from policy')
    except Exception as exc:
        import traceback
        result=dict(status='failed',reason=str(exc),traceback=traceback.format_exc())
    finally:
        builder._run_joint_ik=original;I.JointIKSolver.diagnostic_hook=old_hook
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(result['status'],flush=True)

if __name__=='__main__':main()
