"""CPU-only, isolated development CLI for NCI v2 Steps 3–4."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_devio import (ROOT, MAIN, DEV, Run, canonical, cpu_only, digest,
    load_json, load_surfaces, plain, write_json, write_native_json)
from research.archive.code.interactions.nci_v2.nci_v2_extract import extract_stage, graph_from_record, unique_opportunities
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities

STAGES=('extract','fit-surface','select-motions','measure-response','calibrate')


def validity_adapter(mol):
    """Frozen broad development checks; rigid paths preserve all covalent internals.

    New close approaches must stay above 0.60 covalent-radius sums, or 90% of
    the reference separation when the reference itself is closer. Shell checks
    are exactly the certified pilot's first-shell rules. This is not a complete
    independent stereochemical validator for arbitrary internal-coordinate paths.
    """
    from jacobik.benchmark.coordination import coordination_integrity
    from jacobik.stats.mapping import _covalent_radius
    from jacobik.topology.graph import get_ligand_neighbors
    ref=np.asarray([a.position for a in mol.atoms]); sy=[a.symbol for a in mol.atoms]
    nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
    pos={a.idx:i for i,a in enumerate(mol.atoms)}
    bonded={(min(pos[i],pos[j]),max(pos[i],pos[j])) for i,ns in nb.items() for j in ns}
    pairs=[(i,j) for i in range(len(ref)) for j in range(i+1,len(ref)) if (i,j) not in bonded]
    ii=np.array([a for a,b in pairs]); jj=np.array([b for a,b in pairs])
    r0=np.linalg.norm(ref[ii]-ref[jj],axis=1)
    cut=np.minimum(.60*np.array([_covalent_radius(sy[i])+_covalent_radius(sy[j]) for i,j in pairs]),.90*r0)
    def check(X):
        X=X.detach().numpy(); finite=np.isfinite(X).all((1,2)); ci=coordination_integrity(sy,ref,X)
        d=np.linalg.norm(X[:,ii]-X[:,jj],axis=-1); overlap=(d<cut).any(1)
        bond_ok=np.ones(len(X),dtype=bool)
        for i,j in bonded:
            distance=np.linalg.norm(X[:,i]-X[:,j],axis=-1)
            expected=np.linalg.norm(ref[i]-ref[j])
            bond_ok &= np.abs(distance-expected)<=max(.15,.10*expected)
        return dict(valid=finite&ci['intact']&~overlap&bond_ok,
                    finite=finite,coord_intact=ci['intact'],no_new_gross_overlap=~overlap,
                    ligand_bonds_intact=bond_ok,
                    policy='development_shell_0.30A_bonds_max_0.15A_10percent_clash_0.60_radii')
    return check


def select_stage(spec_path,extract_dir,out=None):
    from research.archive.code.interactions.nci_v2.nci_v2_response import generate_motions
    src=Path(extract_dir); graphfile=src/'opportunities/graphs.json'
    run=Run('select-motions',spec_path,[graphfile,src/'manifest.json'],dict(wider_multiplier=4.,centers=['reference'],include_half_step=True),out)
    if run.resumed: return run.path
    spec=load_json(spec_path); graphs=load_json(graphfile); frame=load_json(src/'frame.json')
    parents={p['code']:p for p in frame['pilot_cohort']['parents']}; all_records={}; selected=0
    for code,raw in graphs.items():
        print('select-motions',code,flush=True)
        mol=graph_from_record(raw); X=np.array([a.position for a in mol.atoms])
        opp=unique_opportunities(enumerate_opportunities(mol,spec))
        records=generate_motions(X,opp,group_id=parents[code]['group_id'],complex_id=code,
                                 mol=mol,schema=spec,validity=validity_adapter(mol),include_half_step=True)
        all_records[code]=records
        selected+=sum(r.get('selected',False) for r in records)
    write_json(run.path/'motions/paths.json',all_records)
    write_json(run.path/'motions/graphs.json',graphs)
    write_json(run.path/'reports/selection.json',dict(submitted=len(graphs),selected=selected,
        attempts=sum(map(len,all_records.values())),energy_calls_during_selection=0,
        per_parent={c:dict(selected=sum(r.get('selected',False) for r in rr),
                            rejected=sum(not r.get('valid',False) for r in rr)) for c,rr in all_records.items()},
        centers=['reference'],development_only=True))
    run.finish(dict(parents=len(graphs),attempted_paths=sum(map(len,all_records.values())),selected_paths=selected),
               ['Conservative rigid-ligand DOFs only; no guessed covalent internals.',
                'Only reference centers wired in this CLI; baseline/trajectory centers require explicit saved solver DOFs.',
                'Independent geometry validator is development-only; fixed stereo is preserved by rigid components.'])
    return run.path


def measure_stage(spec_path,surface_dir,motion_dir,out=None,backend_name='mock',model_path=None,smoke=False):
    from research.archive.code.interactions.nci_v2.nci_v2_response import MockBackend,MACEBackend,measure_motion
    sf=Path(surface_dir)/'surfaces/surfaces.json'; md=Path(motion_dir)
    paths=md/'motions/paths.json'; gf=md/'motions/graphs.json'
    inputs=[sf,paths,gf,Path(surface_dir)/'manifest.json',md/'manifest.json']
    if backend_name=='mace':
        if not smoke: raise ValueError('Real MACE is restricted to the one explicitly requested CPU smoke test')
        if model_path is None: raise ValueError('MACE model required')
        inputs.append(Path(model_path))
    run=Run('measure-response',spec_path,inputs,dict(backend=backend_name,model=str(model_path),
         smoke=smoke,baseline='mock_harmonic_raw_nats_v1',precision='float64'),out)
    if run.resumed: return run.path
    spec=load_json(spec_path); surfaces=load_surfaces(sf); records=load_json(paths); graphs=load_json(gf)
    backend=MockBackend() if backend_name=='mock' else MACEBackend(model_path,max_atoms=70,max_evaluations=30)
    labels=[]; consumed_parents=0; errors=[]
    for code in sorted(graphs,key=lambda c:len(graphs[c]['atoms'])):
        mol=graph_from_record(graphs[code]); sy=[a.symbol for a in mol.atoms]
        opp=unique_opportunities(enumerate_opportunities(mol,spec))
        selected=[r for r in records[code] if r.get('selected') and r.get('valid')]
        if backend_name=='mace':
            if len(sy)>70 or consumed_parents>=1: continue
            selected=selected[:1]
        for motion in selected:
            print('measure-response',backend_name,code,motion['motion_id'],flush=True)
            # The mock baseline is independent of the fitted NCI and explicitly
            # identified in provenance. Actual scorer adapters live in response.py.
            baseline=lambda X: -.01*X.square().sum((1,2))
            try:
                labels.extend(measure_motion(motion,sy,mol.charge,mol.spin_state,baseline,opp,
                                              surfaces,backend,schema=spec,allow_development=True))
            except Exception as exc:
                errors.append(dict(code=code,motion_id=motion['motion_id'],error=f'{type(exc).__name__}: {exc}'))
                if backend_name=='mace': break
        if selected: consumed_parents+=1
    write_json(run.path/'motions/labels.json',labels)
    force=[r['force_fd'] for r in labels if 'force_fd' in r]
    write_json(run.path/'reports/response.json',dict(backend=backend_name,errors=errors,
        label_rows=len(labels),single_points=backend.evaluations,parents=consumed_parents,
        max_projection_fd_error=max((abs(f['error']) for f in force),default=None),
        force_checks=force,full_complex=True,development_mock_baseline=True))
    run.finish(dict(parents=consumed_parents,label_rows=len(labels),single_points=backend.evaluations),
               ['CLI uses a harmonic mock raw baseline, not the frozen production scorer.',
                'MACE smoke validates energy/force plumbing only; these labels must not calibrate a scientific candidate.',
                *[str(e) for e in errors]])
    return run.path


def calibrate_stage(spec_path,surface_dir,response_dir,baseline_db,out=None,bootstrap=200):
    from research.archive.code.interactions.nci_v2.nci_v2_response import fit_response_scales,jsonable,response_diagnostics
    sf=Path(surface_dir)/'surfaces/surfaces.json'; lf=Path(response_dir)/'motions/labels.json'
    run=Run('calibrate',spec_path,[sf,lf,baseline_db,Path(surface_dir)/'manifest.json',Path(response_dir)/'manifest.json'],
            dict(bootstrap=bootstrap,development_only=True),out)
    if run.resumed: return run.path
    spec=load_json(spec_path); surfaces=load_surfaces(sf); labels=load_json(lf)
    response=load_json(Path(response_dir)/'reports/response.json')
    if response['backend']!='mock': raise ValueError('MACE smoke labels cannot calibrate a candidate')
    groups=sorted({r['group_id'] for r in labels})
    artifact=fit_response_scales(surfaces,labels,groups,dict(spec,allow_development=True,bootstrap_samples=bootstrap))
    payload=jsonable(artifact)
    write_json(run.path/'candidate/nci_v2.json',payload)
    raw=load_json(baseline_db); non_nci={k:v for k,v in raw.items() if not k.startswith('NCI:')}
    candidate=dict(non_nci)
    candidate['NCI:v2:development']=dict(schema='nci_v2_steps34_dev_v1',artifact=payload,
        enabled_by_default=False,production_ready=False)
    assert all(candidate[k] is v for k,v in non_nci.items())
    write_native_json(run.path/'candidate/constraints.json',candidate)
    write_json(run.path/'candidate/constraints.meta.json',dict(baseline_sha256=digest(baseline_db),
        non_nci_array_equivalent=True,non_nci_entries=len(non_nci),artifact_sha256=digest(run.path/'candidate/nci_v2.json'),
        identity=run.identity,production_ready=False,certified_runtime_compatible=False))
    write_json(run.path/'reports/calibration.json',payload)
    write_json(run.path/'reports/response_diagnostics.json',response_diagnostics(labels,artifact))
    run.finish(dict(groups=len(groups),label_rows=len(labels),bootstrap=bootstrap),
               ['Development mock labels only. No Gate3 validation labels opened.',
                'Candidate schema is intentionally disabled and requires explicit runtime integration after audit.'])
    return run.path


def main(argv=None):
    cpu_only()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=STAGES)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--frame',type=Path)
    p.add_argument('--pools',type=Path,action='append',default=[])
    p.add_argument('--clean-root',type=Path,default=MAIN/'data/refit/clean')
    p.add_argument('--codes',default='')
    p.add_argument('--development-geometry-matching',action='store_true')
    p.add_argument('--extract-run',type=Path)
    p.add_argument('--surface-run',type=Path)
    p.add_argument('--motion-run',type=Path)
    p.add_argument('--response-run',type=Path)
    p.add_argument('--baseline-db',type=Path)
    p.add_argument('--backend',choices=('mock','mace'),default='mock')
    p.add_argument('--mace-model',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--out-dir',type=Path)
    a=p.parse_args(argv)
    if a.stage=='extract':
        if not a.frame or not a.pools: p.error('extract requires --frame and --pools')
        path=extract_stage(a.spec,a.frame,a.pools,a.clean_root,a.out_dir,a.development_geometry_matching,a.codes.split(',') if a.codes else None)
    elif a.stage=='fit-surface':
        from research.archive.code.interactions.nci_v2.nci_v2_surface import fit_stage
        if not a.extract_run: p.error('--extract-run required')
        path=fit_stage(a.spec,a.extract_run,a.out_dir,a.bootstrap)
    elif a.stage=='select-motions':
        if not a.extract_run:p.error('--extract-run required')
        path=select_stage(a.spec,a.extract_run,a.out_dir)
    elif a.stage=='measure-response':
        if not a.surface_run or not a.motion_run:p.error('--surface-run and --motion-run required')
        path=measure_stage(a.spec,a.surface_run,a.motion_run,a.out_dir,a.backend,a.mace_model,a.smoke)
    else:
        if not a.surface_run or not a.response_run or not a.baseline_db:p.error('--surface-run, --response-run, --baseline-db required')
        path=calibrate_stage(a.spec,a.surface_run,a.response_run,a.baseline_db,a.out_dir,a.bootstrap)
    print(path,flush=True)
    return path


if __name__=='__main__': main()
