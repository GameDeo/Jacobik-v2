"""Reproducible isolated development experiment, serial CPU subprocess stages.

python -m research.archive.code.interactions.nci_v2.nci_v2_dev_run --spec /absolute/pilot_v1/spec.json --mace-smoke
The two requested development panels run separately; no production pools or
calibration/validation energy labels are accessed. At most one child runs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Set limits before numerical imports, also inherited by every child.
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from research.archive.code.interactions.nci_v2.nci_v2_devio import ROOT,MAIN,DEV,Run,load_json,write_json
sys.path.insert(0,str(ROOT/"src"))
os.environ["PYTHONPATH"]=str(ROOT/"src")+os.pathsep+str(ROOT)


def synthetic(spec_path,out=None):
    import numpy as np
    import torch
    from research.archive.code.interactions.nci_v2.nci_v2_surface import fit_density_ratio,bootstrap_surface,latent,support
    from research.archive.code.interactions.nci_v2.nci_v2_response import fit_response_scales,jsonable
    from research.archive.code.interactions.nci_v2.nci_v2 import SurfaceArtifact
    run=Run('synthetic-recovery',spec_path,options=dict(groups=25,alpha=.8,bootstrap=20),out=out)
    if run.resumed:return run.path
    root=('bond_atom','synthetic',('N','H','Cl'),(0,0,0),'single',None,(False,False,False),False,False)
    u=np.linspace(-1,1,35); p=np.exp(.8*u); p/=p.mean(); rows=[]
    for g in range(25):
        for i,t in enumerate(u):
            for y in (0,1):
                rows.append(dict(group_id=str(g),family='bond_atom',context=(root,),features=[4.55+3.45*t,0.],
                                 y=y,weight=float(p[i] if y else 1.)/len(u)))
    spec=load_json(spec_path); groups=[str(g) for g in range(25)]
    start=time.perf_counter();art=fit_density_ratio(rows,groups,spec); fit_wall=time.perf_counter()-start
    X=torch.tensor(np.c_[4.55+3.45*u,np.zeros(len(u))]);s=art.surfaces[str(root)]
    expected=.8*u-np.log(np.exp(.8*u).mean());pred=latent(X,s).detach().numpy()
    start=time.perf_counter();stability=bootstrap_surface(rows,art,groups,spec,20);boot_wall=time.perf_counter()-start
    gamma=2.5
    labels=[dict(group_id=str(g),complex_id=str(g),y=gamma*d,g=gamma*.7*d,
                 D_f={'bond_atom':d},G_f={'bond_atom':.7*d}) for g in range(25) for d in (-2.,-1.,1.,2.)]
    scale_fixture=SurfaceArtifact('bond_atom',{},dict(purpose='independent planted D/G design; no physical surface claim'))
    start=time.perf_counter();cal=fit_response_scales({'bond_atom':scale_fixture},labels,groups,dict(bootstrap_samples=200));cal_wall=time.perf_counter()-start
    zero=fit_response_scales({'bond_atom':scale_fixture},[dict(r,D_f={'bond_atom':0.},G_f={'bond_atom':0.}) for r in labels],groups,dict(bootstrap_samples=20))
    # Separate support challenge: same groups, reference short-range and baseline distant.
    challenge=[dict(group_id=str(g),y=y,weight=1.,features=[r,0.]) for g in range(25) for y,r in ((1,2.),(0,6.))]
    sup=support(challenge,np.array([r['features'] for r in challenge]),[(1.1,8.),(-1.,1.)],spec)
    result=dict(scale_fixture='independent planted D/G design, separate from fitted surface cells',log_ratio_rmse=float(np.sqrt(np.mean((pred-expected)**2))),log_ratio_max_error=float(np.max(np.abs(pred-expected))),
                gamma_planted=gamma,gamma_recovered=cal.gamma_shared,gamma_absolute_error=abs(cal.gamma_shared-gamma),
                zero_design_scale=zero.gamma_shared,zero_design_identified=zero.metadata['identified'],
                support_regimes={reg:int(np.sum(sup['support_regime']==reg)) for reg in ('matched','ref_only','unsupported')},
                surface_fit_wall_s=fit_wall,surface_bootstrap_wall_s=boot_wall,calibration_wall_s=cal_wall,
                work_units=dict(groups=25,rows=len(rows),cv_candidates=96,cv_folds=5,bootstrap_surface=20,bootstrap_scale=200),
                tolerances=dict(log_ratio_rmse=.06,gamma_absolute_error=1e-8),production_ready=False)
    result['passed']=result['log_ratio_rmse']<.06 and result['gamma_absolute_error']<1e-8 and zero.gamma_shared==0. and not zero.metadata['identified']
    write_json(run.path/'reports/recovery.json',result)
    write_json(run.path/'reports/bootstrap.json',stability)
    write_json(run.path/'surfaces/surfaces.json',{'bond_atom':art})
    write_json(run.path/'reports/calibration.json',jsonable(cal))
    run.finish(result['work_units'])
    if not result['passed']:raise AssertionError(result)
    return run.path


def frame_sizes(frame):
    allocation=frame['main_allocation']
    # Stratum split counts are the authoritative allocation, not prose sizes.
    sizes={}
    for s in allocation['strata'].values():
        for key,value in s.get('splits',{}).items():
            sizes[key]=sizes.get(key,0)+int(value)
    return sizes


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--mace-smoke',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--target-fit',type=int,default=1200)
    p.add_argument('--target-calibration',type=int,default=120)
    p.add_argument('--target-validation',type=int,default=120)
    a=p.parse_args(argv)
    if a.synthetic:
        print(synthetic(a.spec,a.out_dir),flush=True);return
    frame_path=MAIN/'data/refit/nci_v2/pilot_v1/frame.json';frame=load_json(frame_path)
    baseline=MAIN/'data/refit/final/constraints_2026-10-18.json'
    run=Run('development',a.spec,[frame_path],dict(mace_smoke=a.mace_smoke,targets=[a.target_fit,a.target_calibration,a.target_validation]),a.out_dir)
    if run.resumed: print(run.path);return
    calls=[]
    def command(stage,args,tag):
        cmd=[sys.executable,'research/archive/code/refit/nci_refit.py',stage,'--spec',str(a.spec),*map(str,args)]
        start=time.perf_counter();res=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
        log=run.path/'reports'/f'{tag}_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True);log.write_text(res.stdout+res.stderr)
        if res.returncode:raise RuntimeError(f'{stage} failed; see {log}')
        path=Path(res.stdout.strip().splitlines()[-1]);manifest=load_json(path/'manifest.json')
        calls.append(dict(panel=tag,stage=stage,path=str(path),process_wall_s=time.perf_counter()-start,
                          stage_wall_s=manifest['wall_s'],peak_rss_mb=manifest['peak_rss_mb'],work_units=manifest['work_units']))
        print(tag,stage,path,flush=True);return path
    panels={}
    for panel,pooldir,codes in (
        ('recert6',MAIN/'data/refit/nci_v2/recert_run_a/pools',frame['pilot_cohort']['repeatability_parents']),
        ('pilot12',MAIN/'data/refit/nci_v2/pilot_v1/pools',[p['code'] for p in frame['pilot_cohort']['parents']])):
        extract=command('extract',['--frame',frame_path,'--pools',pooldir,'--codes',','.join(codes),'--development-geometry-matching'],panel)
        surface=command('fit-surface',['--extract-run',extract,'--bootstrap','20'],panel)
        motion=command('select-motions',['--extract-run',extract],panel)
        response=command('measure-response',['--surface-run',surface,'--motion-run',motion],panel)
        cal=command('calibrate',['--surface-run',surface,'--response-run',response,'--baseline-db',baseline],panel)
        panels[panel]=dict(extract=str(extract),surface=str(surface),motions=str(motion),response=str(response),calibration=str(cal))
    if a.mace_smoke:
        pilot=panels['pilot12']
        smoke=command('measure-response',['--surface-run',pilot['surface'],'--motion-run',pilot['motions'],
             '--backend','mace','--mace-model',MAIN/'tools/jdev/MACE-POLAR-1-S.model','--smoke'],'mace_smoke')
        panels['mace_smoke']=str(smoke)
    # Synthetic work is measured separately; unsupported development roots cannot
    # supply a meaningful production fitting rate.
    cmd=[sys.executable,'-m','research.archive.code.interactions.nci_v2.nci_v2_dev_run','--spec',str(a.spec),'--synthetic']
    res=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
    (run.path/'reports/synthetic.log').write_text(res.stdout+res.stderr)
    if res.returncode:raise RuntimeError('synthetic recovery failed')
    syn=Path(res.stdout.strip().splitlines()[-1]); recovery=load_json(syn/'reports/recovery.json')
    targets=dict(surface_fit=a.target_fit,calibration=a.target_calibration,motion_validation=a.target_validation)
    extrapolation=[]
    for c in calls:
        n=c['work_units'].get('parents',c['work_units'].get('groups',0))
        if c['stage']=='fit-surface' or not n:continue
        target=a.target_fit if c['stage']=='extract' else a.target_calibration
        extrapolation.append(dict(panel=c['panel'],stage=c['stage'],observed_units=n,
            unit='parent or independent group',seconds_per_unit=c['stage_wall_s']/n,
            target_units=target,projected_wall_s=c['stage_wall_s']/n*target,
            quadratic_upper_s=c['stage_wall_s']*(target/n)**2 if c['stage']=='calibrate' else None,
            assumed='same parent sizes, opportunity counts, paths and backend as observed; linear work scaling',
            memory_projection='not linear; observed RSS only, batch/stream implementation dependent'))
    fit_projection=dict(observed_groups=25,target_groups=a.target_fit,
        observed_rows=recovery['work_units']['rows'],observed_cv_fits=480,
        quadratic_upper_wall_s=recovery['surface_fit_wall_s']*(a.target_fit/25)**2,
        surface_fit_wall_s=recovery['surface_fit_wall_s'],
        illustrative_one_root_linear_wall_s=recovery['surface_fit_wall_s']*a.target_fit/25,
        limits='Synthetic one-root 2D rate. Production root/context distribution, dimensions and row counts unknown; this is not a total runtime estimate.')
    write_json(run.path/'reports/development.json',dict(panels=panels,stages=calls,frame_sizes=frame_sizes(frame),
        target_sizes=targets,extrapolation=extrapolation,fit_projection=fit_projection,synthetic=str(syn),recovery=recovery,
        limitations=['No production or untouched validation results.','No supported roots in the small development panels.',
                     'No measured real-MACE throughput beyond the one bounded smoke.','Validation scaling uses the same geometry/label procedure but no validation labels are opened.'],
        production_ready=False))
    run.finish(dict(panels=2,stages=len(calls)))
    print(run.path,flush=True)


if __name__=='__main__':main()
