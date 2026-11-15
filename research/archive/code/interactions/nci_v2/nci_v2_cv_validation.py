"""Full exact/binned development CV and float64 training-input replay."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from research.archive.code.interactions.nci_v2.nci_v2_columnar import RootRows,write_columns,support_mass,PreparedRows,ChunkedProblem,grouped_cv_root
from research.archive.code.interactions.nci_v2.nci_v2_columnar_bench import ROOT
from research.archive.code.interactions.nci_v2.nci_v2_binning import bin_root,supported_error
from research.archive.code.interactions.nci_v2.nci_v2_devio import Run,write_json,load_json
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget


def norm_coeff(s):
    return float(np.max(np.abs(s['coef_r'])))+sum(float(np.max(np.abs(x))) for x in s['coef_rx'])


def coeff_delta(a,b):
    return float(np.max(np.abs(np.asarray(a['coef_r'])-b['coef_r'])))+sum(float(np.max(np.abs(np.asarray(x)-y))) for x,y in zip(a['coef_rx'],b['coef_rx']))


def fit_delta_bound(a,b):
    """Global convex B-spline bound, also allowing different context masks."""
    bound=coeff_delta(a,b);aa=a.get('context_corrections',{});bb=b.get('context_corrections',{})
    for key in set(aa)|set(bb):
        x=aa.get(key);y=bb.get(key)
        if x is None:bound+=norm_coeff(y);continue
        if y is None:bound+=norm_coeff(x);continue
        dm=float(np.max(np.abs(np.asarray(x['support_a'])-y['support_a'])))
        bound+=coeff_delta(x,y)+dm*norm_coeff(y)
    return bound


def derivative_coeff_bound(c,knots,axis):
    c=np.asarray(c);n=c.shape[axis]
    if n<2:return 0.
    den=np.asarray(knots)[4:4+n-1]-np.asarray(knots)[1:1+n-1]
    shape=[1]*c.ndim;shape[axis]=n-1
    return float(np.max(np.abs(3*np.diff(c,axis=axis)/den.reshape(shape))))


def component_gradient_bound(s):
    bs=s['basis_specs'];n=len(bs['knots_x'])+1;g=np.zeros(n)
    g[0]=derivative_coeff_bound(s['coef_r'],bs['knots_r'],0)
    for j,c in enumerate(s['coef_rx'],1):
        g[0]+=derivative_coeff_bound(c,bs['knots_r'],0)
        g[j]+=derivative_coeff_bound(c,bs['knots_x'][j-1],1)
    return g


def coordinate_rounding_bound(s,domains):
    g=component_gradient_bound(s)
    for ct in s.get('context_corrections',{}).values():
        spec=ct['support_basis_spec'];knots=[spec['knots_r'],*spec['knots_x']]
        mgrad=np.array([derivative_coeff_bound(ct['support_a'],k,j) for j,k in enumerate(knots)])
        g+=component_gradient_bound(dict(ct,basis_specs=s['basis_specs']))+mgrad*norm_coeff(ct)
    eps=np.array([float(np.spacing(np.float32(max(abs(lo),abs(hi)))))/2 for lo,hi in domains])
    return dict(max_feature_rounding=eps.tolist(),gradient_bound=g.tolist(),global_abs_delta_F=float(g@eps))


class Float64Replay:
    def __init__(self,rows,X):self.__dict__.update(rows.__dict__);self.exact_X=X
    def coordinates(self,sl):return self.exact_X[sl]
    slices=RootRows.slices


def run(spec_path,out):
    budget(start=True);spec=load_json(spec_path);cfg=spec['cv_validation'];n=cfg['physical_rows'];groups=cfg['groups']
    if n%2 or n//2%groups:raise ValueError('paired subset requires equal groups')
    r=Run('followup-cv-choice',spec_path,[],dict(subset='first paired feature points of prior seed17 synthetic stream before float32 conversion',physical_rows=n,groups=groups),out)
    if r.resumed:return r.path
    rng=np.random.default_rng(17);u=rng.random((n//2,4));X=np.c_[1.1+6.9*u[:,0],2*u[:,1]-1,2*u[:,2]-1,u[:,3]]
    raw=[];lookup={}
    for i,x in enumerate(X):
        g=str(i%groups);ctx=(ROOT,(ROOT,2),((ROOT,2),'Fe' if i%groups%2 else 'Co','octahedral'))
        for y in (0,1):
            w=float(np.exp(.6*u[i,0]+.1*u[i,1])) if y else 1.
            raw.append(dict(group_id=g,context=ctx,family='bond_atom',features=x.tolist(),y=y,weight=w))
            key=(g,y,tuple(x.astype(np.float32)))
            if key in lookup:raise ValueError('subset float32 collision needs a declared merge replay')
            lookup[key]=x
    write_columns(iter(raw),r.path/'columns',spec);rows=RootRows(r.path/'columns')
    exactX=np.array([lookup[(rows.dataset['groups'][int(g)],int(y),tuple(x))] for g,y,x in zip(rows.g,rows.y,rows.x[:,:4])])
    np.savez(r.path/'input_float64.npz',X=exactX,g=rows.g,y=rows.y,w=rows.w)
    book=support_mass(rows,spec);print('exact CV',n,'rows',len(rows.group_names),'groups',flush=True)
    t=time.perf_counter();exact,a=grouped_cv_root(rows,spec,mass_cache=book);exact_s=time.perf_counter()-t
    write_json(r.path/'reports/exact_cv.json',a);write_json(r.path/'surfaces/exact.json',exact)
    bin_root(rows,r.path/'binned',spec,cfg['radial_bins'],cfg['angular_bins']);summary=RootRows(r.path/'binned')
    print('binned CV',summary.n,'summaries',flush=True)
    t=time.perf_counter();approx,b=grouped_cv_root(summary,spec,mass_cache=book);approx_s=time.perf_counter()-t
    write_json(r.path/'reports/binned_cv.json',b);write_json(r.path/'surfaces/binned.json',approx)
    # Replay the selected exact configuration with float64 input AND support.
    replay=Float64Replay(rows,exactX);t=time.perf_counter();p=ChunkedProblem(PreparedRows(replay,spec),a['selected']['config']);f=p.fit()
    if not f.success:raise RuntimeError('float64 replay nonconverged')
    double=p.surface(f.x);qtime=time.perf_counter()-t
    write_json(r.path/'surfaces/float64.json',double)
    lossesA=[v['mean'] for v in a['results']];lossesB=[v['mean'] for v in b['results']]
    binerr=supported_error(rows,exact,approx,seed=23,count=4096);qerr=supported_error(rows,exact,double,seed=23,count=4096)
    bound=fit_delta_bound(exact,double);rounding=coordinate_rounding_bound(exact,rows.meta['domains'])
    report=dict(physical_rows=n,stored_rows=rows.n,independent_groups=len(rows.group_names),summary_rows=summary.n,
        full_grid_configurations=96,folds=5,exact_wall_s=exact_s,binned_wall_s=approx_s,float64_replay_wall_s=qtime,
        exact_one_se_choice=a['selected']['config'],binned_one_se_choice=b['selected']['config'],
        same_one_se_choice=a['selected']['config']==b['selected']['config'],spearman_cv_loss=float(spearmanr(lossesA,lossesB).statistic),
        max_cv_loss_delta=float(np.max(np.abs(np.array(lossesA)-lossesB))),binned_error=binerr,
        float32_quantization=dict(error=qerr,global_model_difference_bound=bound,coordinate_rounding=rounding,
            combined_global_bound=bound+rounding['global_abs_delta_F'],support_mask_changed_cells=int(np.count_nonzero(np.asarray(exact['support_a'])!=double['support_a'])),
            replay_config=a['selected']['config'],scope='selected exact config; full float64 CV not performed'),
        exact_executed_fits=sum(x.get('fits',0) for x in a['execution']),binned_executed_fits=sum(x.get('fits',0) for x in b['execution']))
    write_json(r.path/'reports/comparison.json',report)
    r.finish(dict(physical_rows=n,groups=groups,full_grid_cv_runs=2),['Synthetic subset only; not a real-pool or production generalization bound.'])
    return r.path


def run_double_cv(spec_path,source_run,out):
    budget(start=True);spec=load_json(spec_path);source=Path(source_run)
    inputs=[source/'manifest.json',source/'input_float64.npz',source/'reports/exact_cv.json',source/'columns/dataset.json',*source.glob('columns/root_*.npz')]
    r=Run('followup-float64-full-cv',spec_path,inputs,dict(source_run=str(source),full_cv=True),out)
    if r.resumed:return r.path
    rows=RootRows(source/'columns')
    with np.load(source/'input_float64.npz',allow_pickle=False) as z:
        for key in ('g','y','w'):
            if not np.array_equal(z[key],getattr(rows,key)):raise ValueError('float64 replay source-row identity mismatch')
        replay=Float64Replay(rows,z['X'])
    t=time.perf_counter();surface,cv=grouped_cv_root(replay,spec);elapsed=time.perf_counter()-t
    previous=load_json(source/'reports/exact_cv.json');a=np.array([x['mean'] for x in previous['results']]);b=np.array([x['mean'] for x in cv['results']])
    report=dict(physical_rows=rows.n,independent_groups=len(rows.group_names),configurations=96,folds=5,
        float32_choice=previous['selected']['config'],float64_choice=cv['selected']['config'],same_one_se_choice=previous['selected']['config']==cv['selected']['config'],
        spearman_cv_loss=float(spearmanr(a,b).statistic),max_cv_loss_delta=float(np.max(np.abs(a-b))),wall_s=elapsed)
    write_json(r.path/'reports/comparison.json',report);write_json(r.path/'reports/float64_cv.json',cv);write_json(r.path/'surfaces/float64.json',surface)
    r.finish(dict(rows=rows.n,full_cv_runs=1),['Same synthetic subset; finite-precision numerical replay, not production data.'])
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-run');a=p.parse_args();print(run_double_cv(a.spec,a.replay_run,a.out) if a.replay_run else run(a.spec,a.out))
