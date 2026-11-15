"""Complete available native gradients/endpoints when a common-reference path fails."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tools.jgen.g3e_native_twist import ROOT,PROTOCOL,load_capture,vector,evaluate,write_json
from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.core.env import MACE_MODEL_PATH


def complete(unit,stats,mace):
    d=ROOT/f"{unit['code']}_{unit['seed']}";path=json.loads((d/'path.json').read_text())
    if path['status']=='ok':
        if not (d/'gradients.json').exists():write_json(d/'gradients.json',dict(status='ok',observations=path['native_observations']))
        return mace
    if all((d/f).exists() for f in ['gradients.json','endpoint_mace.json','reference_diagnostic.json']):return mace
    selected=json.loads((d/'selection.json').read_text());cap,meta=load_capture(d,'R2s',stats);observations={};positions={}
    for role in ['twisted','top1','untwisted']:
        identity=selected[role];assert identity['arm']=='R2s'
        row=meta['rows'][identity['index']];trial=row['key'][2];value,X=evaluate(cap,vector(cap,trial),trial)
        residual=float(np.sqrt(np.mean(np.sum((X-np.asarray(row['X_A']))**2,axis=1))));error=value['score_native']-row['score']
        if residual>1e-5 or abs(error)>1e-4:raise ValueError(f'native verification failed for {role}: {residual}, {error}')
        observations[role]=dict(value,score_error=error,reconstruction_rmsd_A=residual,key=row['key']);positions[role]=X
    write_json(d/'gradients.json',dict(status='ok',observations=observations))
    tw=selected['twisted']['key'][2];ut=selected['untwisted']['key'][2];v=vector(cap,ut)
    P,Q=positions['untwisted'],positions['twisted'];heavy=[i for i,s in enumerate(meta['symbols']) if s!='H']
    u,_,vt=np.linalg.svd((P[heavy]-P[heavy].mean(0)).T@(Q[heavy]-Q[heavy].mean(0)));D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);G=u@D@vt
    ref=cap.X_full.double();ref=ref if ref.ndim==2 else ref[ut];offset=0
    for fk,theta in zip(cap.fks,cap.thetas_final):
        n=theta.shape[1];pivot=ref[fk._donor_pos_t].mean(0).numpy() if fk._donor_pos_t.numel() else ref.mean(0).numpy()
        v[offset:offset+3]=(pivot+v[offset:offset+3])@G-pivot
        v[offset+3:offset+6]=Rotation.from_matrix(G.T@Rotation.from_rotvec(v[offset+3:offset+6]).as_matrix()).as_rotvec();offset+=n
    _,own=evaluate(cap,v,ut);_,common=evaluate(cap,v,tw)
    write_json(d/'reference_diagnostic.json',dict(unit=unit,
        own_reference_aligned_RMSD_A=float(np.sqrt(np.mean(np.sum((own-P@G)**2,axis=1)))),
        common_reference_RMSD_A=float(np.sqrt(np.mean(np.sum((common-P@G)**2,axis=1)))),
        reference_shape=list(cap.X_full.shape),macro_joints=[f.n_macro_joints for f in cap.fks],
        explanation='Different sampled FK references; no alternative partner or nonlinear path substituted.'))
    if mace is None:
        from jacobik.core.energy import MACEPolarEvaluator
        mace=MACEPolarEvaluator(model='polar-1-s',device='cpu',dtype='float64',mode='energy-only',threads=1,model_path=str(MACE_MODEL_PATH)).load()
    energy=mace.energies([(meta['symbols'],positions[role],meta['charge'],meta['spin']) for role in ['twisted','untwisted']])
    write_json(d/'endpoint_mace.json',dict(roles=['twisted','untwisted'],energy_kj=(np.asarray(energy)*96.4853321233).tolist(),path_minimum_unavailable=True))
    return mace


def main():
    torch.set_num_threads(1);stats=stats_from_db(inputs.DB_PATH);mace=None
    for unit in json.loads(PROTOCOL.read_text())['units']:
        try:mace=complete(unit,stats,mace)
        except Exception as exc:
            write_json(ROOT/f"{unit['code']}_{unit['seed']}"/'components_failure.json',dict(unit=unit,status='failed',reason=str(exc)))
        print(unit,flush=True)

if __name__=='__main__':main()
