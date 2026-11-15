"""Grouped density-ratio fitting and smooth chemistry-preserving partial pooling.

This development artifact has an explicit evaluator; the certified v2 runtime is
unchanged and does not yet execute spatial context corrections (see design note).
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
import copy
import hashlib
import json
import math
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit
import torch
from research.archive.code.interactions.nci_v2.nci_v2 import (SurfaceArtifact, FAMILIES, DEFAULT_RADIAL_WINDOWS,
    compute_tapers, h_epsilon, features_torch, evaluate_latent_surface, evaluate_joint_support)
from research.archive.code.interactions.nci_v2.nci_v2_extract import freeze
from research.archive.code.interactions.nci_v2.nci_v2_devio import Run, load_json, load_surfaces, plain, write_json


def knots(domain, interiors):
    lo,hi = domain
    return np.r_[np.repeat(lo,4), np.linspace(lo,hi,interiors+2)[1:-1], np.repeat(hi,4)]


def domains(family, root, dim, spec):
    h = family=='bond_atom' and (len(root)<3 or root[2][1]=='H')
    key = f"bond_atom_{'H' if h else 'heavy'}" if family=='bond_atom' else family
    win = DEFAULT_RADIAL_WINDOWS[key]
    declared = spec.get('radial_windows',{}).get(family,{})
    if family=='bond_atom': declared=declared.get('h_endpoint' if h else 'non_h_endpoint',{})
    if 'window' in declared: win=tuple(declared['window'])
    dom = [(win[0],win[3])]
    for k in range(1,dim):
        if family=='bond_atom': dom.append((0.,1.) if k==3 else (-1.,1.))
        elif family=='bond_ring': dom.append((-1.,1.) if k==1 else (0.,1.))
        else: dom.append((0.,2.) if k==2 else (0.,1.))
    return dom, win


def basis(x, kk):
    return BSpline.design_matrix(np.clip(x,kk[0],kk[-1]),kk,3).toarray()


def joint_values(X, sp, grid):
    bs = [basis(X[:,k],kk) for k,kk in enumerate([sp['knots_r']]+sp['knots_x'])]
    letters='abcd'[:X.shape[1]]
    expr=','.join('n'+s for s in letters)+','+letters+'->n'
    return np.einsum(expr,*bs,grid,optimize=True)


def support(rows, X, dom, spec):
    cfg=spec.get('joint_support_grid',{})
    # Resolution is intervals in the roadmap; cubic bases = intervals + 3.
    ri=int(cfg.get('radial_intervals',6)); ai=int(cfg.get('angular_intervals',3))
    sp=dict(knots_r=knots(dom[0],ri-1),knots_x=[knots(d,ai-1) for d in dom[1:]])
    kk=[sp['knots_r']]+sp['knots_x']; shape=tuple(len(k)-4 for k in kk)
    groups=sorted({r['group_id'] for r in rows}); gi={g:i for i,g in enumerate(groups)}
    v=np.zeros((2,len(groups),*shape)); w=np.array([r.get('weight',1.) for r in rows])
    bs=[basis(X[:,k],knot) for k,knot in enumerate(kk)]
    letters='abcd'[:X.shape[1]]
    expr='n,'+','.join('n'+s for s in letters)+'->'+letters
    for source in (0,1):
        for group in groups:
            idx=np.array([i for i,r in enumerate(rows) if r['group_id']==group and r['y']==source],dtype=int)
            if len(idx):
                v[source,gi[group]]=np.einsum(expr,w[idx],*[b[idx] for b in bs],optimize=True)
    tau=1e-3*np.mean([sum(r.get('weight',1.) for r in rows if r['group_id']==g) for g in groups])
    def stats(a):
        mass=a.sum(0); sq=(a*a).sum(0)
        return (a>tau).sum(0),np.divide(mass*mass,sq,out=np.zeros_like(mass),where=sq>0)
    nb,eb=stats(v[0]); nr,er=stats(v[1]); npool,ep=stats(v.sum(0))
    good=(nr>=10)&(er>=10-1e-10); matched=good&(nb>=10)&(eb>=10-1e-10)
    root_good=all(len({r['group_id'] for r in rows if r['y']==s})>=20 for s in (0,1))
    ro=good&~matched&root_good
    a=(matched|ro).astype(float)
    return dict(support_basis_spec=sp,support_a=a,ref_only_mask=ro,
                support_regime=np.where(matched,'matched',np.where(ro,'ref_only','unsupported')),
                counts_ref=nr,ess_ref=er,counts_base=nb,ess_base=eb,
                counts_pool=npool,ess_pool=ep,tau_grp=float(tau))


def design(X, dom, interior):
    # Knot mapping standardizes coordinates to [0,1] for the regularizer. Store
    # equivalent physical knots so both evaluators use the same basis functions.
    kk=[knots(d,interior) for d in dom]
    bs=[basis(X[:,k],knot) for k,knot in enumerate(kk)]
    transforms=[]; blocks=[bs[0]]; sizes=[(bs[0].shape[1],)]
    for k,b in enumerate(bs[1:],1):
        count=b.shape[1]
        # Uniform-domain integrals center interactions; drop one column to remove
        # the duplicate radial main effect exactly (not only empirically).
        integral=(kk[k][4:]-kk[k][:-4])/(4*(dom[k][1]-dom[k][0]))
        H=np.eye(count)[:-1]-integral[:-1,None]
        transforms.append(H)
        bx=b@H.T
        blocks.append((bs[0][:,:,None]*bx[:,None,:]).reshape(len(X),-1))
        sizes.append((bs[0].shape[1],count-1))
    phi=np.concatenate(blocks,axis=1)
    # Integrated squared second derivatives in normalized coordinates, computed
    # by Gaussian quadrature. Row-major tensor coefficient order is explicit.
    def matrices(knot,domain):
        kn=(knot-domain[0])/(domain[1]-domain[0]); n=len(kn)-4
        spline=BSpline(kn,np.eye(n),3)
        gx,gw=np.polynomial.legendre.leggauss(4)
        M=np.zeros((n,n)); R=M.copy()
        for a,b in zip(np.unique(kn)[:-1],np.unique(kn)[1:]):
            t=a+(gx+1)*(b-a)/2; w=gw*(b-a)/2
            b0=spline(t); b2=spline(t,nu=2)
            M+=(b0.T*w)@b0; R+=(b2.T*w)@b2
        return M,R
    mr,rr=matrices(kk[0],dom[0]); total=phi.shape[1]; penalty=np.zeros((total,total))
    penalty[:len(mr),:len(mr)]=rr
    off=len(mr)
    for j,H in enumerate(transforms,1):
        mx,rx=matrices(kk[j],dom[j]); mx=H@mx@H.T; rx=H@rx@H.T
        p=np.kron(rr,mx)+np.kron(mr,rx); size=len(p)
        penalty[off:off+size,off:off+size]=p; off+=size
    return phi,penalty,dict(knots_r=kk[0],knots_x=kk[1:]),sizes,transforms


def unpack(beta, sizes, transforms):
    off=sizes[0][0]; cr=beta[:off]; cx=[]
    for shape,H in zip(sizes[1:],transforms):
        n=math.prod(shape); cx.append(beta[off:off+n].reshape(shape)@H); off+=n
    return cr,cx


def _fit_root(rows,family,root,spec,config):
    X=np.asarray([r['features'] for r in rows]); y=np.array([r['y'] for r in rows],float)
    w=np.array([r.get('weight',1.) for r in rows],float)
    if not np.isfinite(X).all() or np.any(w<=0) or not np.isin(y,[0,1]).all():
        raise ValueError('invalid density rows')
    if not all(np.any(y==s) for s in (0,1)): raise ValueError('both sources required')
    w=np.array([a/(2*w[y==yy].sum()) for a,yy in zip(w,y)])
    dom,win=domains(family,root,X.shape[1],spec)
    phi,R,bs,sizes,H=design(X,dom,config['interior_knots']); p=phi.shape[1]
    sup=support(rows,X,dom,spec)
    contexts=[freeze(r['context']) for r in rows]
    corrections={}; parts=[phi]
    # Corrections at every supported descendant, with smooth reliability. There
    # are no ancestors that drop charge/binding/root chemistry.
    for level in (1,2):
        for ctx in sorted({c[level] for c in contexts if len(c)>level},key=str):
            idx=np.array([i for i,c in enumerate(contexts) if len(c)>level and c[level]==ctx],dtype=int)
            subrows=[rows[i] for i in idx]
            if not all(len({r['group_id'] for r in subrows if r['y']==s})>=10 for s in (0,1)):
                continue
            ss=support(subrows,X[idx],dom,spec)
            reliability=np.zeros(len(rows)); reliability[idx]=joint_values(X[idx],ss['support_basis_spec'],ss['support_a'])
            if not np.any(reliability): continue
            corrections[str(ctx)]=dict(level=level,**ss)
            parts.append(phi*reliability[:,None])
    A=np.concatenate(parts,axis=1)
    def objective(beta):
        z=A@beta; loss=np.dot(w,np.logaddexp(0,z)-y*z); grad=A.T@(w*(expit(z)-y))
        for j in range(len(parts)):
            b=beta[j*p:(j+1)*p]
            ridge=config['lambda_0'] if j==0 else config['lambda_z']
            # Mean coefficient ridge and integrated normalized-coordinate roughness.
            reg=config['lambda_r']*(R@b)+ridge*b
            loss+=b@reg; grad[j*p:(j+1)*p]+=2*reg
        return loss,grad
    res=minimize(objective,np.zeros(A.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=500,ftol=1e-10,gtol=1e-7,maxcor=15))
    cr,cx=unpack(res.x[:p],sizes,H)
    surf=dict(root_key=root,coef_r=cr,coef_rx=cx,basis_specs=bs,radial_window=win,
              feature_dimension=X.shape[1],**sup,context_corrections=corrections,
              convergence=dict(success=bool(res.success),message=str(res.message),iterations=res.nit,gradient_max=float(np.max(np.abs(res.jac)))))
    for j,c in enumerate(corrections.values(),1):
        c['coef_r'],c['coef_rx']=unpack(res.x[j*p:(j+1)*p],sizes,H)
    return surf


def tensor_surface(s,X,cache=None):
    key=id(s)
    if cache is not None and key in cache:return cache[key]
    def spec(d): return {k:[torch.as_tensor(a,dtype=X.dtype,device=X.device) for a in v] if k=='knots_x' else torch.as_tensor(v,dtype=X.dtype,device=X.device) for k,v in d.items()}
    result=dict(coef_r=torch.as_tensor(s['coef_r'],dtype=X.dtype,device=X.device),
                coef_rx=[torch.as_tensor(v,dtype=X.dtype,device=X.device) for v in s['coef_rx']],
                basis_specs=spec(s['basis_specs']),support_basis_spec=spec(s['support_basis_spec']),
                support_a=torch.as_tensor(s['support_a'],dtype=X.dtype,device=X.device))
    if cache is not None:cache[key]=result
    return result


def latent(X,s,context=None,tensor_cache=None):
    ts=tensor_surface(s,X,tensor_cache)
    val=evaluate_latent_surface(X,ts['basis_specs'],ts['coef_r'],ts['coef_rx'])
    if context is not None:
        for key in map(str,freeze(context)[1:]):
            c=s.get('context_corrections',{}).get(key)
            if c is None: continue
            ct=tensor_surface(dict(c,basis_specs=s['basis_specs']),X)
            m=evaluate_joint_support(X,ct['support_basis_spec'],ct['support_a'])
            val=val+m*evaluate_latent_surface(X,ts['basis_specs'],ct['coef_r'],ct['coef_rx'])
    return val


def _predict(rows, artifact):
    result=np.zeros(len(rows))
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[freeze(r['context'])].append(i)
    for ctx,idx in groups.items():
        s=artifact.surfaces.get(str(ctx[0]))
        if s is not None:
            X=torch.tensor([rows[i]['features'] for i in idx],dtype=torch.float64)
            result[idx]=latent(X,s,ctx).detach().numpy()
    return result


def expanded_views(rows):
    """One physical row remains one parent mass; internal symmetric views share it."""
    out=[]
    for r in rows:
        views=r.get('feature_views',[r['features']])
        for view in views:
            item=dict(r,features=view,weight=r.get('weight',1.)/len(views))
            item.pop('feature_views',None)
            out.append(item)
    return out


def fit_fixed(rows,groups,spec,config,minimum_root_groups=20,diagnose_unsupported=True):
    valid=expanded_views([r for r in rows if r['group_id'] in set(groups)])
    fam=rows[0]['family'] if rows else 'bond_atom'; by=defaultdict(list)
    for r in valid: by[freeze(r['context'])[0]].append(r)
    surfaces={}; unsupported=[]
    for root,rr in sorted(by.items(),key=lambda kv:str(kv[0])):
        counts={str(s):len({r['group_id'] for r in rr if r['y']==s}) for s in (0,1)}
        if min(counts.values())<minimum_root_groups:
            entry=dict(root=str(root),groups=counts,reason='fewer_than_20_groups_in_either_source')
            if diagnose_unsupported:
                X=np.array([r['features'] for r in rr]);dom,_=domains(fam,root,X.shape[1],spec)
                diag=support(rr,X,dom,spec)
                diag['support_a']=np.zeros_like(diag['support_a'])
                diag['ref_only_mask']=np.zeros_like(diag['ref_only_mask'])
                diag['support_regime']=np.full(diag['support_a'].shape,'unsupported')
                entry['joint_support']=diag
            unsupported.append(entry);continue
        surfaces[str(root)]=_fit_root(rr,fam,root,spec,config)
    return SurfaceArtifact(fam,surfaces,plain(spec),dict(config=config,unsupported_roots=unsupported))


def grouped_folds(groups,k=5):
    # Hash order is independent of labels and stable under row/parent order.
    groups=sorted(set(groups),key=lambda g:hashlib.sha256(str(g).encode()).hexdigest())
    return [groups[i::k] for i in range(k)]


def logloss(rows,pred):
    losses=[]
    for y in (0,1):
        idx=[i for i,r in enumerate(rows) if r['y']==y]
        if not idx: return None
        w=np.array([rows[i].get('weight',1.) for i in idx]); z=pred[idx]
        losses.append(np.dot(w,np.logaddexp(0,z)-y*z)/w.sum()/2)
    return float(sum(losses))


def fit_density_ratio(rows,fit_split,spec=None):
    sp=spec or {}; rows=expanded_views([r for r in rows if r['group_id'] in set(fit_split)])
    grid=[dict(interior_knots=n,lambda_r=lr,lambda_z=lz,lambda_0=l0)
          for n,lr,lz,l0 in product((4,6),(.01,.1,1.,10.),(.01,.1,1.,10.),(.001,.01,.1))]
    folds=grouped_folds(fit_split); report=[]
    rootcounts=defaultdict(lambda:[set(),set()])
    for r in rows: rootcounts[freeze(r['context'])[0]][r['y']].add(r['group_id'])
    if not any(min(map(len,v))>=20 for v in rootcounts.values()):
        cfg=dict(interior_knots=4,lambda_r=10.,lambda_z=10.,lambda_0=.1)
        art=fit_fixed(rows,fit_split,sp,cfg)
        art.metadata.update(cv=dict(status='no_supported_roots',requested_folds=5,candidate_count=len(grid),evaluated_candidates=0,
                                    reason='Every root fails mandatory independent-group threshold; no parameters can be selected.'))
        return art
    if min(map(len,folds))==0: raise ValueError('five grouped folds require five groups')
    for cfg in grid:
        losses=[]
        for hold in folds:
            train=set(fit_split)-set(hold); valid=[r for r in rows if r['group_id'] in hold]
            art=fit_fixed(rows,train,sp,cfg,diagnose_unsupported=False)
            loss=logloss(valid,_predict(valid,art))
            if loss is not None: losses.append(loss)
        mean=float(np.mean(losses)); se=float(np.std(losses,ddof=1)/np.sqrt(len(losses)))
        report.append(dict(config=cfg,fold_losses=losses,mean=mean,se=se))
    best=min(report,key=lambda r:r['mean']); eligible=[r for r in report if r['mean']<=best['mean']+best['se']]
    # Predeclared simplicity: fewer knots, larger context penalty, root ridge,
    # then smoother surface. Lexicographic ordering is explicit and deterministic.
    chosen=min(eligible,key=lambda r:(r['config']['interior_knots'],-r['config']['lambda_z'],-r['config']['lambda_0'],-r['config']['lambda_r']))
    art=fit_fixed(rows,fit_split,sp,chosen['config'])
    art.metadata['cv']=dict(status='complete',folds=folds,candidate_count=len(grid),results=report,
                            best=best,selected=chosen,rule='simplest_within_best_plus_one_standard_error')
    return art


def occurrence_rewards(X,opportunities,surfaces,schema=None,ref_only=False):
    if X.ndim==2: X=X[None]
    fb=features_torch(X,opportunities,schema)
    reverse_opportunities=copy.deepcopy(opportunities)
    has_symmetric=False
    for candidates in reverse_opportunities.opportunities.values():
        for o in candidates:
            if o.get('symmetric_bond'):
                o['pos_A'],o['pos_B']=o['pos_B'],o['pos_A'];has_symmetric=True
    reversed_features=features_torch(X,reverse_opportunities,schema) if has_symmetric else fb
    result={}; tensor_cache={}
    for fam in FAMILIES:
        occs=opportunities.opportunities[fam]; values=[None]*len(occs); artifact=surfaces.get(fam)
        grouped=defaultdict(list)
        for i,o in enumerate(occs):
            root=freeze(o['z_o'])[0]
            grouped[(str(root),repr(freeze(o['z_o'])),bool(o.get('symmetric_bond')))].append(i)
        for (root,_context,_symmetric),indices in grouped.items():
            o=occs[indices[0]]
            s=artifact.surfaces.get(str(freeze(o['z_o'])[0])) if artifact else None
            if s is None:
                for i in indices:values[i]=X.sum((1,2))*0
                continue
            def value(features):
                x=features[:,:, :s['feature_dimension']].reshape(-1,s['feature_dimension'])
                ts=tensor_surface(s,x,tensor_cache); a=ts['support_a']
                if ref_only: a=a*torch.as_tensor(s['ref_only_mask'],dtype=x.dtype,device=x.device)
                m=evaluate_joint_support(x,ts['support_basis_spec'],a)
                tin,tout=compute_tapers(x[:,0],*s['radial_window'])
                return (tin*tout*m*h_epsilon(latent(x,s,o['z_o'],tensor_cache))).reshape(len(X),len(indices))
            selected=fb.features[fam][:,indices,:]
            reward=value(selected)
            if _symmetric:
                reward=.5*(reward+value(reversed_features.features[fam][:,indices,:]))
            for j,i in enumerate(indices):values[i]=reward[:,j]
        result[fam]=torch.stack(values,1) if values else X[:, :0, 0]
    return result


def bootstrap_surface(rows,artifact,groups,spec,repeats=20):
    rng=np.random.default_rng(0); groups=sorted(set(groups)); predictions=[]
    probes=rows[:min(len(rows),512)]
    for b in range(repeats):
        sample=rng.choice(groups,len(groups),replace=True); rr=[]
        for i,g in enumerate(sample):
            rr.extend(dict(r,group_id=f'bootstrap_{i}') for r in rows if r['group_id']==g)
        fit=fit_fixed(rr,[f'bootstrap_{i}' for i in range(len(sample))],spec,artifact.metadata['config'],diagnose_unsupported=False)
        predictions.append(_predict(probes,fit))
    arr=np.asarray(predictions)
    return dict(seed=0,replicates=repeats,probe_rows=len(probes),
                standard_deviation=np.std(arr,axis=0,ddof=1).tolist() if repeats>1 else [],
                interpretation='group-resampled pointwise stability; no simultaneous confidence claim')


def read_fit_rows(path):
    """Stream JSONL and intern repeated contexts; retain only estimator columns."""
    contexts={};rows=[]
    with open(path) as f:
        for line in f:
            r=json.loads(line);ctx=freeze(r['context']);ctx=contexts.setdefault(ctx,ctx)
            item={k:r[k] for k in ('family','features','weight','group_id','y')}
            item['context']=ctx
            if len(r.get('feature_views',[]))>1:item['feature_views']=r['feature_views']
            rows.append(item)
    return rows


def fit_stage(spec_path,extract_dir,out=None,bootstrap=20):
    from pathlib import Path
    src=Path(extract_dir); rowpath=src/'opportunities/rows.jsonl'
    run=Run('fit-surface',spec_path,[rowpath,src/'manifest.json'],dict(bootstrap=bootstrap),out)
    if run.resumed: return run.path
    rows=read_fit_rows(rowpath); spec=load_json(spec_path)
    surfaces={}; stability={}
    for fam in FAMILIES:
        rr=[r for r in rows if r['family']==fam]; groups=sorted({r['group_id'] for r in rr})
        art=fit_density_ratio(rr,groups,spec); art.family=fam; surfaces[fam]=art
        stability[fam]=bootstrap_surface(rr,art,groups,spec,bootstrap) if art.surfaces else dict(status='no_supported_roots')
    write_json(run.path/'surfaces/surfaces.json',surfaces)
    write_json(run.path/'reports/cv.json',{f:a.metadata for f,a in surfaces.items()})
    write_json(run.path/'reports/bootstrap.json',stability)
    plot_diagnostics(rows,surfaces,run.path/'reports/surface_support.png')
    run.finish(dict(rows=len(rows),roots=sum(len(a.surfaces) for a in surfaces.values()),
                    requested_cv_fits=96*5,bootstrap=bootstrap),
               ['Certified runtime cannot yet consume spatial context corrections; new evaluator is development-only.',
                'Small pilot cannot satisfy 20 independent groups per root; an empty surface is a valid development outcome.'])
    return run.path


def plot_diagnostics(rows,surfaces,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,9))
    for j,fam in enumerate(FAMILIES):
        rr=[r for r in rows if r['family']==fam]
        for y,color in ((1,'tab:blue'),(0,'tab:orange')):
            xx=np.array([r['features'] for r in rr if r['y']==y],dtype=object)
            if len(xx): axes[j,0].scatter([v[0] for v in xx],[v[1] for v in xx],s=1,alpha=.2,c=color,label=str(y))
        axes[j,0].set(title=f'{fam}: sampled r / orientation',xlabel='r (Å)',ylabel='orientation')
        art=surfaces[fam]
        if art.surfaces:
            s=next(iter(art.surfaces.values())); dim=s['feature_dimension']
            x=np.zeros((100,dim)); x[:,0]=np.linspace(s['radial_window'][0],s['radial_window'][-1],100)
            axes[j,1].plot(x[:,0],latent(torch.tensor(x),s).detach().numpy(),label='latent slice, other coordinates=0')
            axes[j,1].plot(x[:,0],joint_values(x,s['support_basis_spec'],np.asarray(s['support_a'])),label='support')
            axes[j,1].legend(fontsize=6)
        else: axes[j,1].text(.1,.5,'No root with ≥20 groups in both sources',transform=axes[j,1].transAxes)
    fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
