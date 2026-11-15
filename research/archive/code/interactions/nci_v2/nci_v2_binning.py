"""Optional, declared weighted feature binning for the fit loss only.

Source columns and exact independent-group support masses remain authoritative.
Fold summaries never supply group counts or parent attribution. Full grouped CV
is an explicit option, deferred while S2 runs under the agreed compute regime.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,ast,copy
from pathlib import Path
import time
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_columnar import (RootRows,DTYPE,PreparedRows,ChunkedProblem,
    support_mass,grouped_cv_root,host_budget)
from research.archive.code.interactions.nci_v2.nci_v2_surface import grouped_folds,latent,joint_values
from research.archive.code.interactions.nci_v2.nci_v2_devio import Run,write_json,load_json,digest
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import BudgetPause

torch.set_num_threads(1)
BIN_POLICY=dict(version='fit-loss-binning.1',radial_bins=8,angular_bins=4,
    feature_representative='source-weighted mean within fold/context/source/bin',
    grouping='predeclared group folds; no cross-fold or cross-chemistry merging',
    exact_support='original per-group masses, counts and ESS, rebuilt within folds',
    parent_attribution='must use original columns, never fit-only summaries',
    approximation_error='sampled max absolute delta F on original and random supported probes',
    cv_comparison='same 96 configurations and 5 original-group folds; deferred until S2 completes')


def bin_root(rows,out,spec,radial_bins=8,angular_bins=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);host_budget()
    if rows.dataset.get('fit_only_summary'):raise ValueError('cannot rebin an approximate summary')
    n=rows.n;dim=rows.dim
    folds=grouped_folds(rows.dataset['groups']);fold_map={g:i for i,ff in enumerate(folds) for g in ff}
    original_fold=np.array([fold_map[g] for g in rows.group_names],dtype=np.uint8)[rows.group_index]
    counts=[radial_bins]+[angular_bins]*(dim-1)
    # Keep independent group identity in the bin key. This preserves exact
    # group-held-out folds and group-level support during binned CV.
    dtype=np.dtype([('g','<u4'),('c2','<u4'),('c3','<u4'),('y','u1'),('clip','u1'),('bins','<u2',(4,))])
    key=np.zeros(n,dtype=dtype);key['g']=rows.g;key['c2']=rows.c2;key['c3']=rows.c3;key['y']=rows.y;key['clip']=rows.clip
    for sl in rows.slices(32768):
        X=rows.coordinates(sl)
        for j,((lo,hi),bins) in enumerate(zip(rows.meta['domains'],counts)):
            key['bins'][sl,j]=np.minimum(bins-1,np.maximum(0,np.floor((X[:,j]-lo)/(hi-lo)*bins))).astype(np.uint16)
    host_budget();unique,inv=np.unique(key,return_inverse=True);host_budget()
    weight=np.bincount(inv,weights=rows.w);result=np.zeros(len(unique),dtype=DTYPE)
    for field in ('c2','c3','y','clip'):result[field]=unique[field]
    result['dim']=dim;result['w']=weight
    result['multiplicity']=np.bincount(inv,weights=rows.multiplicity).astype(np.uint32)
    result['g']=unique['g']
    representatives={int(rows.g[i]):int(rows.parent[i]) for i in np.unique(rows.g,return_index=True)[1]}
    result['parent']=[representatives[int(g)] for g in unique['g']]
    for j in range(dim):
        sums=np.zeros(len(unique))
        for sl in rows.slices(32768):
            sums+=np.bincount(inv[sl],weights=rows.w[sl]*rows.coordinates(sl)[:,j],minlength=len(unique))
        result['x'][:,j]=(sums/weight).astype(np.float32)
    path=out/'root_0000.npz';np.savez(path,**{name:result[name] for name in DTYPE.names},root_id=np.array(0,dtype=np.uint32))
    meta=copy.deepcopy(rows.dataset);r=copy.deepcopy(rows.meta);r.update(id=0,file=path.name,rows=len(result),sha256=digest(path))
    meta['roots']=[r];meta['fit_only_summary']=True;meta['fit_folds']=folds
    meta['approximation']=dict(BIN_POLICY,radial_bins=radial_bins,angular_bins=angular_bins,source_rows=n,summary_rows=len(result),
        source_dataset_sha256=digest(rows.directory/'dataset.json'),source_root_sha256=rows.meta['sha256'])
    write_json(out/'dataset.json',meta)
    return meta


def supported_error(source,exact_surface,approx_surface,seed=0,count=4096):
    rng=np.random.default_rng(seed);idx=np.linspace(0,source.n-1,min(count,source.n),dtype=int)
    observed=source.coordinates(idx);random=np.column_stack([rng.uniform(a,b,count) for a,b in source.meta['domains']])
    X=np.r_[observed,random];gate=joint_values(X,exact_surface['support_basis_spec'],np.asarray(exact_surface['support_a']))
    supported=gate>=.5;labels=source.dataset['contexts'];root=ast.literal_eval(source.meta['root'])
    context_pairs=set();errors=[]
    for sl in source.slices(65536):
        context_pairs.update(map(tuple,np.unique(np.c_[source.c2[sl],source.c3[sl]],axis=0)))
    with torch.no_grad():
        for c2,c3 in sorted(context_pairs):
            ctx=[root]
            for c in (c2,c3):
                if c:ctx.append(ast.literal_eval(labels[int(c)]))
            xx=torch.tensor(X,dtype=torch.float64)
            a=latent(xx,exact_surface,ctx).numpy();b=latent(xx,approx_surface,ctx).numpy()
            if np.any(supported):errors.extend(np.abs(a[supported]-b[supported]).tolist())
    return dict(probe_seed=seed,probe_count=len(X),supported_probe_count=int(supported.sum()),
        contexts=len(context_pairs),max_abs_delta_F=max(errors) if errors else None,
        rms_delta_F=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        caveat='Sampled maximum over declared probes, not an analytic supremum.')


def run(spec_path,columns,out,radial=8,angular=4,full_cv=False):
    if os.nice(0)<19:raise ValueError('nice -n 19 required')
    host_budget(start=True);spec=load_json(spec_path)
    inputs=[Path(columns)/'dataset.json',*sorted(Path(columns).glob('root_*.npz'))]
    run=Run('phase2-binning-comparison',spec_path,inputs,dict(radial_bins=radial,angular_bins=angular,full_cv=full_cv,policy=BIN_POLICY),out)
    if run.resumed:return run.path
    original=RootRows(columns);t=time.perf_counter();book=support_mass(original,spec);support_s=time.perf_counter()-t
    t=time.perf_counter();meta=bin_root(original,run.path/'binned',spec,radial,angular);binning_s=time.perf_counter()-t
    cfg=dict(interior_knots=4,lambda_r=.1,lambda_z=.1,lambda_0=.01)
    t=time.perf_counter();exact_problem=ChunkedProblem(PreparedRows(original,spec,mass_cache=book),cfg);exact_fit=exact_problem.fit();exact_s=time.perf_counter()-t
    exact_surface=exact_problem.surface(exact_fit.x)
    summary=RootRows(run.path/'binned');t=time.perf_counter();approx_problem=ChunkedProblem(PreparedRows(summary,spec,mass_cache=book),cfg);approx_fit=approx_problem.fit();approx_s=time.perf_counter()-t
    approximate_surface=approx_problem.surface(approx_fit.x)
    if not exact_fit.success or not approx_fit.success:raise RuntimeError('comparison requires converged fits')
    error=supported_error(original,exact_surface,approximate_surface)
    cv=dict(status='deferred_until_S2_completes')
    if full_cv:
        _,a=grouped_cv_root(original,spec,mass_cache=book);_,b=grouped_cv_root(summary,spec,mass_cache=book)
        cv=dict(status='complete',exact_selected=a['selected'],approximate_selected=b['selected'],same_choice=a['selected']['config']==b['selected']['config'])
        write_json(run.path/'reports/exact_cv.json',a);write_json(run.path/'reports/approximate_cv.json',b)
    report=dict(source_rows=original.n,summary_rows=summary.n,row_reduction=original.n/summary.n,
        support_wall_s=support_s,binning_wall_s=binning_s,exact_fit_wall_s=exact_s,approximate_fit_wall_s=approx_s,
        original_independent_groups=len(original.group_names),summary_group_labels=len(summary.group_names),
        exact_support_group_counts=approx_problem.prepared.root_counts.tolist(),
        raw_weight_original=float(original.w.sum()),raw_weight_summary=float(summary.w.sum()),error=error,cv=cv,
        approximation_policy=meta['approximation'])
    write_json(run.path/'reports/comparison.json',report)
    write_json(run.path/'surfaces/exact.json',exact_surface);write_json(run.path/'surfaces/approximate.json',approximate_surface)
    run.finish(dict(source_rows=original.n,summary_rows=summary.n,full_cv=full_cv),
        ['Approximate fit loss only. Original source rows remain necessary for group bootstrap and attribution.',
         'Exact independent-group support is used; summary fold representatives are not independent groups.',
         'CV-choice comparison is deferred unless explicitly requested after S2.'])
    return run.path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--columns',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--radial-bins',type=int,default=8);p.add_argument('--angular-bins',type=int,default=4);p.add_argument('--full-cv',action='store_true');a=p.parse_args()
    try:print(run(a.spec,a.columns,a.out_dir,a.radial_bins,a.angular_bins,a.full_cv),flush=True)
    except BudgetPause as e:print(f'PAUSED {e}',flush=True);raise SystemExit(75)

if __name__=='__main__':main()
