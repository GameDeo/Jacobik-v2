"""Stored coordination availability and unconditional geometric pose coverage."""
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget,health_metrics
from research.archive.code.interactions.nci_v2.nci_v2_extract import graph_from_record
from research.archive.code.interactions.nci_v2.nci_v2_poolio import mapped_geometry
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,Run,load_json,write_json
from jacobik.benchmark.coordination import coordination_integrity, COORD_SHIFT_TOL_A, COORD_BOUND_SCALE
import argparse,ast,hashlib,inspect,re,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def strata(parent):return '_'.join(str(parent.get(k,'unknown')) for k in ('metal_row','charge_class','cn_class'))


def coordination_record(z,X):
    ci=coordination_integrity(z['ref_symbols'],z['ref_X'],X)
    available='coord_intact' in z.files
    record=dict(poses=len(X),stored_available=available,recomputed_loss=int((~ci['intact']).sum()),
        stored_loss=int((~z['coord_intact'].astype(bool)).sum()) if available else None,
        flag_disagreements=int((z['coord_intact'].astype(bool)!=ci['intact']).sum()) if available else None)
    record['stored_shift_max_delta']={k:float(np.max(np.abs(z[k]-ci[k]))) for k in ('sigma_shift_A','haptic_slip_A','haptic_shift_A') if k in z.files}
    return record,ci


def run(spec_path,frame_path,graph_path,out):
    budget(start=True);spec=load_json(spec_path);policy=spec['solver_artifact_controls'];frame=load_json(frame_path);graphs=load_json(graph_path)
    parents={p['code']:p for p in frame['pilot_cohort']['parents']};paths={}
    for panel,part in [('pilot','pilot_v1/pools'),('recert','recert_run_a/pools')]:
        paths[panel]=[p for p in sorted((MAIN/'data/refit/nci_v2'/part).glob('*.npz')) if p.name.split('_')[0] in parents and
            (re.fullmatch(r'.+_s[0-9]+\.npz',p.name) if panel=='pilot' else p.name.endswith('_control.npz'))]
    r=Run('followup-coordinate-health',spec_path,[frame_path,graph_path,*sum(paths.values(),[])],dict(all_conformers=True,pose_dedup=False,energy_access=False),out)
    if r.resumed:return r.path
    result={}
    for panel,pp in paths.items():
        overall=Counter();by=defaultdict(Counter);groupids=defaultdict(set);records=[]
        for path in pp:
            budget();code=path.name.split('_')[0];parent=parents[code];name=strata(parent);mol=graph_from_record(graphs[code]);ref=np.asarray([a.position for a in mol.atoms])
            with np.load(path,allow_pickle=False) as z:
                if 'error' in z.files:continue
                X,info=mapped_geometry(z,pool_path=path)
                if not np.array_equal(z['ref_symbols'],[a.symbol for a in mol.atoms]) or not np.allclose(z['ref_X'],ref,atol=1e-7):raise ValueError('reference identity mismatch')
                record,ci=coordination_record(z,X);record.update(pool=path.name,code=code,stratum=name)
                stored=z['coord_intact'] if record['stored_available'] else None
                health,_,_,_=health_metrics(mol,ref,X,policy,stored)
                selfhealth,_,_,_=health_metrics(mol,ref,ref[None],policy)
                record['reference_health_pass']=bool(selfhealth['healthy'][0]);record['removed_health']=int((~health['healthy']).sum())
                delta=dict(pools=1,poses=len(X),removed_coord=record['recomputed_loss'],removed_health=record['removed_health'],
                    stored_available_pools=int(record['stored_available']),stored_available_poses=len(X) if record['stored_available'] else 0,
                    stored_missing_pools=int(not record['stored_available']),stored_missing_poses=0 if record['stored_available'] else len(X),
                    stored_loss=record['stored_loss'] or 0,flag_disagreements=record['flag_disagreements'] or 0,
                    recomputed_loss_where_stored_available=record['recomputed_loss'] if record['stored_available'] else 0,
                    recomputed_loss_where_stored_missing=0 if record['stored_available'] else record['recomputed_loss'])
                overall.update(delta);by[name].update(delta);groupids[name].add(parent['group_id']);records.append(record)
        groups=[]
        for name,values in sorted(by.items()):groups.append(dict(stratum=name,**values,independent_groups=len(groupids[name]),removed_coord_fraction=values['removed_coord']/values['poses'],removed_health_fraction=values['removed_health']/values['poses']))
        result[panel]=dict(counts=dict(overall),strata=groups,pools=records)
        print(panel,dict(overall),flush=True)
    sources={}
    for rev in ['85b57f7','951cfcb']:
        source=subprocess.check_output(['git','show',rev+':research/archive/code/refit/nci_v2_pilot.py'],text=True)
        tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordination_integrity')
        sources[rev]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    write_json(r.path/'reports/coordination_health.json',dict(panels=result,function_ast_sha256=sources,same_function_at_certified_commits=len(set(sources.values()))==1,
        constants=dict(shift_tol_A=COORD_SHIFT_TOL_A,bound_scale=COORD_BOUND_SCALE),threshold_basis=policy['threshold_basis']))
    r.finish(dict(panels=2,pool_files=sum(map(len,paths.values()))),['Pose-removal fractions are raw conformer fractions, not independent-group or parent-balanced fit weights.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--graphs',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(run(a.spec,a.frame,a.graphs,a.out))
