"""Bounded-memory columnar rows and exact (same-precision) chunked spline loss.

No out-of-domain row is discarded: certified clamped splines give such rows a
boundary contribution. Canonical boundary pseudo-observations retain that mass.
Features are float32 by declared storage contract; weights remain float64.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
from collections import defaultdict
import heapq
import json
from itertools import product
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from research.archive.code.interactions.nci_v2.nci_v2_devio import DEV,write_json,load_json,digest
from research.archive.code.interactions.nci_v2.nci_v2_extract import freeze
from research.archive.code.interactions.nci_v2.nci_v2_surface import domains,design,basis,unpack,knots,joint_values,grouped_folds
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import host_budget as _host_budget

def host_budget(start=False):
    available=float(os.environ.get('NCI_V2_MIN_AVAILABLE_GIB',6. if start else 5.))
    max_rss=float(os.environ.get('NCI_V2_MAX_RSS_MIB',1280.))
    return _host_budget(min_available_gib=available,max_rss_mib=max_rss)

# Root id is in the file/dictionary; context ids are stable within this dataset.
DTYPE=np.dtype([('parent','<u4'),('g','<u4'),('c2','<u4'),('c3','<u4'),('y','u1'),('dim','u1'),
                ('clip','u1'),('x','<f4',(4,)),('w','<f8'),('multiplicity','<u4')])
KEYTYPE=np.dtype([(k,DTYPE.fields[k][0]) for k in ('parent','g','c2','c3','y','dim','clip','x')])


def allowed(path):
    path=Path(path)
    if not path.resolve().is_relative_to(DEV.resolve()):raise ValueError('output must be under data_dev')
    path.mkdir(parents=True,exist_ok=True)
    return path


def record_key(r):
    return tuple(int(r[k]) for k in ('parent','g','c2','c3','y','dim','clip'))+tuple(float(v) for v in r['x'])


def merge_chunk(a):
    key=np.empty(len(a),dtype=KEYTYPE)
    for k in KEYTYPE.names:key[k]=a[k]
    _,idx,inv=np.unique(key,return_index=True,return_inverse=True)
    out=a[idx].copy()
    out['w']=np.bincount(inv,weights=a['w'])
    out['multiplicity']=np.bincount(inv,weights=a['multiplicity']).astype(np.uint32)
    return out


def write_columns(rows,out,spec,chunk_size=32768):
    """Stream physical rows/views; sort/merge bounded runs, then a heap merge.

    No Python object collection grows with input N. Equal group/context/source
    and exactly identical stored features share a sufficient weighted row.
    """
    out=allowed(out)
    if (out/'dataset.json').exists():raise ValueError('columnar dataset is immutable')
    rootids={};roots=[];groupids={};parentids={};contexts={};context_labels=['None'];buffers=[]
    parts=defaultdict(list);counts=defaultdict(lambda:dict(input_views=0,weight=0.,outside_weight=0.))
    def intern(mapping,label):
        if label not in mapping:mapping[label]=len(mapping)
        return mapping[label]
    def context_id(ctx):
        if ctx is None:return 0
        key=str(freeze(ctx))
        if key not in contexts:contexts[key]=len(context_labels);context_labels.append(key)
        return contexts[key]
    def flush():
        if not buffers:return
        host_budget()
        by=defaultdict(list)
        for rid,r in buffers:by[rid].append(r)
        for rid,items in by.items():
            aa=merge_chunk(np.array(items,dtype=DTYPE));p=out/f'r{rid:04d}_part{len(parts[rid]):05d}.npy'
            np.save(p,aa,allow_pickle=False);parts[rid].append(p)
        buffers.clear()
    physical=0
    for row in rows:
        physical+=1
        ctx=freeze(row['context']);root=ctx[0];key=(row['family'],str(root),len(row['features']))
        if key not in rootids:
            rid=len(roots);rootids[key]=rid
            dom,win=domains(row['family'],root,key[2],spec)
            roots.append(dict(id=rid,family=row['family'],root=str(root),dimension=key[2],domains=dom,radial_window=win))
        rid=rootids[key];meta=roots[rid];dim=meta['dimension']
        gid=intern(groupids,str(row['group_id']));pid=intern(parentids,str(row.get('parent_id',row['group_id'])));c2=context_id(ctx[1] if len(ctx)>1 else None);c3=context_id(ctx[2] if len(ctx)>2 else None)
        views=row.get('feature_views',[row['features']]);weight=float(row.get('weight',1.))/len(views)
        if not np.isfinite(weight) or weight<=0 or row['y'] not in (0,1):raise ValueError('invalid source/weight')
        for view in views:
            x=np.zeros(4,dtype=np.float32);x[:dim]=view
            if not np.isfinite(x).all():raise ValueError('nonfinite feature')
            clip=0
            for j,(lo,hi) in enumerate(meta['domains']):
                if float(x[j])<lo:clip|=1<<(2*j);x[j]=lo
                elif float(x[j])>hi:clip|=2<<(2*j);x[j]=hi
            x[x==0]=0 # signed-zero equivalence
            buffers.append((rid,(pid,gid,c2,c3,int(row['y']),dim,clip,x,weight,1)))
            counts[rid]['input_views']+=1;counts[rid]['weight']+=weight
            if clip&3:counts[rid]['outside_weight']+=weight
            if len(buffers)>=chunk_size:flush()
    flush()
    for rid,pp in sorted(parts.items()):
        host_budget();runs=[np.load(p,mmap_mode='r') for p in pp];heap=[]
        for i,a in enumerate(runs):
            if len(a):heapq.heappush(heap,(record_key(a[0]),i,0))
        raw=out/f'r{rid:04d}.bin';pending=[];previous=None;weight=0.;mult=0;representative=None;n=0
        with raw.open('wb') as stream:
            def emit():
                nonlocal n
                if representative is None:return
                item=representative.copy();item['w']=weight;item['multiplicity']=mult;pending.append(item);n+=1
                if len(pending)>=chunk_size:
                    np.asarray(pending,dtype=DTYPE).tofile(stream);pending.clear();host_budget()
            while heap:
                key,i,j=heapq.heappop(heap);r=runs[i][j]
                if key!=previous:
                    emit();previous=key;representative=r;weight=0.;mult=0
                weight+=float(r['w']);mult+=int(r['multiplicity'])
                j+=1
                if j<len(runs[i]):heapq.heappush(heap,(record_key(runs[i][j]),i,j))
            emit()
            if pending:np.asarray(pending,dtype=DTYPE).tofile(stream)
        data=np.memmap(raw,dtype=DTYPE,mode='r',shape=(n,));target=out/f'root_{rid:04d}.npz'
        np.savez(target,**{name:data[name] for name in DTYPE.names},root_id=np.array(rid,dtype=np.uint32))
        del data,runs
        raw.unlink()
        for p in pp:p.unlink()
        roots[rid].update(file=target.name,rows=n,sha256=digest(target),**counts[rid])
    metadata=dict(schema='nci_v2_columnar_v1',roots=roots,groups=list(groupids),contexts=context_labels,
                  parents=list(parentids),physical_input_rows=physical,feature_dtype='float32',weight_dtype='float64',
                  reduction='exact duplicate merging at stored precision; boundary mass retained, never deleted')
    write_json(out/'dataset.json',metadata)
    return metadata


class RootRows:
    def __init__(self,directory,root_id=0,verify=True):
        self.directory=Path(directory);self.dataset=load_json(self.directory/'dataset.json');self.meta=self.dataset['roots'][root_id]
        path=self.directory/self.meta['file']
        if verify and digest(path)!=self.meta['sha256']:raise ValueError('column checksum mismatch')
        with np.load(path,allow_pickle=False) as z:
            for name in DTYPE.names:
                setattr(self,name,z[name] if name in z.files else z['g'] if name=='parent' else None)
        self.dataset.setdefault('parents',self.dataset['groups'])
        self.n=len(self.w);self.dim=self.meta['dimension'];self.spec={}
        unique,inv=np.unique(self.g,return_inverse=True);self.unique_groups=unique;self.group_index=inv.astype(np.int32)
        self.group_names=[self.dataset['groups'][int(i)] for i in unique]
    def coordinates(self,sl):
        x=self.x[sl,:self.dim].astype(np.float64)
        flags=self.clip[sl]
        for j,(lo,hi) in enumerate(self.meta['domains']):
            x[(flags>>(2*j)&3)==1,j]=lo;x[(flags>>(2*j)&3)==2,j]=hi
        return x
    def slices(self,chunk=8192):
        for i in range(0,self.n,chunk):host_budget();yield slice(i,min(i+chunk,self.n))
    def legacy_rows(self):
        # Small-data reference/testing adapter only, never called by the fit.
        root=ast.literal_eval(self.meta['root']);labels=self.dataset['contexts'];out=[]
        X=self.coordinates(slice(0,self.n))
        for i in range(self.n):
            ctx=[root]
            for key in ('c2','c3'):
                ident=int(getattr(self,key)[i])
                if ident:ctx.append(ast.literal_eval(labels[ident]))
            out.append(dict(features=X[i].tolist(),context=tuple(ctx),family=self.meta['family'],
                            group_id=self.dataset['groups'][int(self.g[i])],y=int(self.y[i]),weight=float(self.w[i])))
        return out


class SplineBasis:
    def __init__(self,domains,interiors):
        _,self.R,self.spec,self.sizes,self.H=design(np.zeros((1,len(domains))),domains,interiors)
        self.p=len(self.R)
    def phi(self,X):
        br=basis(X[:,0],self.spec['knots_r']);blocks=[br]
        for j,(kk,H) in enumerate(zip(self.spec['knots_x'],self.H),1):
            bx=basis(X[:,j],kk)@H.T
            blocks.append((br[:,:,None]*bx[:,None,:]).reshape(len(X),-1))
        return np.concatenate(blocks,axis=1)


def support_mass(rows,spec,chunk=1024):
    """Exact group/source joint masses, using only the 4**D local basis entries."""
    dom=rows.meta['domains'];cfg=spec.get('joint_support_grid',{})
    kk=[knots(dom[0],int(cfg.get('radial_intervals',6))-1)]+[knots(d,int(cfg.get('angular_intervals',3))-1) for d in dom[1:]]
    shape=tuple(len(k)-4 for k in kk);cells=int(np.prod(shape));G=len(rows.unique_groups)
    v=np.zeros((2,G,cells));context_mass={};pair_maps={}
    for level in ('c2','c3'):
        ids=getattr(rows,level)
        if not np.any(ids):continue
        pairs,inv=np.unique(np.c_[rows.group_index,ids],axis=0,return_inverse=True)
        pair_maps[level]=(pairs,inv.astype(np.int32));context_mass[level]=np.zeros((2,len(pairs),cells))
    for sl in rows.slices(chunk):
        X=rows.coordinates(sl);indices=np.zeros((len(X),1),dtype=np.int32);values=np.ones((len(X),1))
        for j,k in enumerate(kk):
            b=__import__('scipy').interpolate.BSpline.design_matrix(np.clip(X[:,j],k[0],k[-1]),k,3)
            col=b.indices.reshape(len(X),4);val=b.data.reshape(len(X),4)
            indices=(indices[:,:,None]*shape[j]+col[:,None,:]).reshape(len(X),-1)
            values=(values[:,:,None]*val[:,None,:]).reshape(len(X),-1)
        src=rows.y[sl].astype(np.int64);weights=rows.w[sl,None]*values
        flat=(src[:,None]*G+rows.group_index[sl,None])*cells+indices
        np.add.at(v.reshape(-1),flat.reshape(-1),weights.reshape(-1))
        for level,a in context_mass.items():
            inv=pair_maps[level][1][sl];flat=(src[:,None]*a.shape[1]+inv[:,None])*cells+indices
            np.add.at(a.reshape(-1),flat.reshape(-1),weights.reshape(-1))
    return dict(knots_r=kk[0],knots_x=kk[1:]),shape,v,context_mass,pair_maps,rows.group_names


def support_from_mass(v,shape):
    # v[source,independent-group,cell], with zero/nontraining groups removed.
    mass=v.sum((0,2));active=mass>0;v=v[:,active,:]
    tau=.001*float(v.sum())/max(v.shape[1],1)
    N=(v>tau).sum(1);s=v.sum(1);sq=(v*v).sum(1)
    ess=np.divide(s*s,sq,out=np.zeros_like(s),where=sq>0)
    root_counts=(v.sum(2)>0).sum(1);matched=(N[1]>=10)&(ess[1]>=10-1e-10)&(N[0]>=10)&(ess[0]>=10-1e-10)
    ref=(N[1]>=10)&(ess[1]>=10-1e-10)&~matched&bool(np.min(root_counts)>=20)
    return dict(support_a=(matched|ref).reshape(shape).astype(float),ref_only_mask=ref.reshape(shape),
                counts_ref=N[1].reshape(shape),ess_ref=ess[1].reshape(shape),
                counts_base=N[0].reshape(shape),ess_base=ess[0].reshape(shape),tau_grp=tau),root_counts


class PreparedRows:
    def __init__(self,rows,spec,heldout=(),mass_cache=None):
        self.rows=rows;self.spec=spec;self.heldout=set(heldout)
        self.train_groups=np.array([name not in self.heldout for name in rows.group_names])
        self.train=self.train_groups[rows.group_index]
        totals=[rows.w[self.train&(rows.y==y)].sum() for y in (0,1)]
        if min(totals)<=0:raise ValueError('training fold requires both sources')
        self.weight=np.where(self.train,rows.w/(2*np.asarray(totals)[rows.y]),0.)
        sp,shape,v,contexts,pair_maps,support_names=mass_cache or support_mass(rows,spec)
        support_train=np.array([name not in self.heldout for name in support_names])
        self.support_spec=sp;self.support,self.root_counts=support_from_mass(v[:,support_train],shape)
        self.corrections=[];self.reliability=[];self.block_ids=[]
        for level in ('c2','c3'):
            ids=getattr(rows,level);blocks=np.zeros(rows.n,dtype=np.int32);reliable=np.zeros(rows.n)
            if level in contexts:
                pairs,_=pair_maps[level]
                eligible_ids=sorted(set(ids)-{0},key=lambda c:rows.dataset['contexts'][int(c)])
                for cid in eligible_ids:
                    idx=np.flatnonzero((pairs[:,1]==cid)&support_train[pairs[:,0]])
                    a=contexts[level][:,idx,:];sup,counts=support_from_mass(a,shape)
                    if np.min(counts)<10 or not np.any(sup['support_a']):continue
                    block=len(self.corrections)+1
                    self.corrections.append(dict(level=level,context_id=int(cid),support=sup))
                    for sl in rows.slices(4096):
                        local=np.flatnonzero(ids[sl]==cid)
                        if not len(local):continue
                        xx=rows.coordinates(sl)[local]
                        values=joint_values(xx,sp,sup['support_a'])
                        reliable[sl.start+local]=values;blocks[sl.start+local]=block
            self.reliability.append(reliable);self.block_ids.append(blocks)


class ChunkedProblem:
    def __init__(self,prepared,config,chunk=8192):
        self.prepared=prepared;self.config=config;self.chunk=chunk
        self.basis=SplineBasis(prepared.rows.meta['domains'],config['interior_knots'])
        self.p=self.basis.p;self.blocks=1+len(prepared.corrections)
    def logits(self,beta,sl,phi=None):
        rr=self.prepared.rows;P=self.p
        phi=self.basis.phi(rr.coordinates(sl)) if phi is None else phi
        z=phi@beta[:P]
        for ids,m in zip(self.prepared.block_ids,self.prepared.reliability):
            bid=ids[sl];mm=m[sl]
            for b in np.unique(bid):
                if not b:continue
                idx=np.flatnonzero(bid==b);z[idx]+=mm[idx]*(phi[idx]@beta[b*P:(b+1)*P])
        return z
    def objective(self,beta):
        rows=self.prepared.rows;loss=0.;grad=np.zeros_like(beta);P=self.p
        for sl in rows.slices(self.chunk):
            phi=self.basis.phi(rows.coordinates(sl));z=self.logits(beta,sl,phi);w=self.prepared.weight[sl]
            loss+=float(w@(np.logaddexp(0,z)-rows.y[sl]*z));res=w*(expit(z)-rows.y[sl])
            grad[:P]+=phi.T@res
            for ids,m in zip(self.prepared.block_ids,self.prepared.reliability):
                bid=ids[sl];mm=m[sl]
                for b in np.unique(bid):
                    if not b:continue
                    idx=np.flatnonzero(bid==b);grad[b*P:(b+1)*P]+=phi[idx].T@(res[idx]*mm[idx])
        for b in range(self.blocks):
            bb=beta[b*P:(b+1)*P];ridge=self.config['lambda_0'] if b==0 else self.config['lambda_z']
            reg=self.config['lambda_r']*(self.basis.R@bb)+ridge*bb
            loss+=float(bb@reg);grad[b*P:(b+1)*P]+=2*reg
        return loss,grad
    def strict_polish(self,beta,steps=6):
        """Small-system Newton polish separates optimizer tolerance from algebra."""
        from scipy.linalg import solve
        size=len(beta)
        if size>1024:raise ValueError('strict Newton polish limited to 1024 coefficients; use chunked L-BFGS for large systems')
        rows=self.prepared.rows;P=self.p
        for _ in range(steps):
            value,gradient=self.objective(beta)
            if np.max(np.abs(gradient))<2e-12:break
            H=np.zeros((size,size))
            for sl in rows.slices(self.chunk):
                phi=self.basis.phi(rows.coordinates(sl));parts=[phi]
                for b in range(1,self.blocks):
                    m=sum(np.where(ids[sl]==b,rel[sl],0.) for ids,rel in zip(self.prepared.block_ids,self.prepared.reliability))
                    parts.append(phi*m[:,None])
                A=np.concatenate(parts,axis=1);prob=expit(A@beta)
                H+=(A.T*(self.prepared.weight[sl]*prob*(1-prob)))@A
            for b in range(self.blocks):
                sl=slice(b*P,(b+1)*P);ridge=self.config['lambda_0'] if b==0 else self.config['lambda_z']
                H[sl,sl]+=2*(self.config['lambda_r']*self.basis.R+ridge*np.eye(P))
            delta=solve(H,gradient,assume_a='pos');beta=beta-delta
        return beta
    def fit(self,start=None,strict=False,maxiter=2000,precondition=True):
        from scipy.linalg import cholesky,solve_triangular
        P=self.p;size=P*self.blocks
        beta=np.zeros(size) if start is None else start
        # Exact change of coordinates: each penalty block becomes alpha.T alpha.
        # No dense N-by-P design, Hessian approximation, or estimator change.
        factors=[]
        for b in range(self.blocks):
            ridge=self.config['lambda_0'] if b==0 else self.config['lambda_z']
            factors.append(cholesky(self.config['lambda_r']*self.basis.R+ridge*np.eye(P),lower=True))
        def to_beta(alpha):
            if not precondition:return alpha
            return np.concatenate([solve_triangular(L,alpha[b*P:(b+1)*P],lower=True,trans='T',check_finite=False) for b,L in enumerate(factors)])
        def transformed(alpha):
            value,gradient=self.objective(to_beta(alpha))
            if precondition:gradient=np.concatenate([solve_triangular(L,gradient[b*P:(b+1)*P],lower=True,check_finite=False) for b,L in enumerate(factors)])
            return value,gradient
        initial=np.concatenate([L.T@beta[b*P:(b+1)*P] for b,L in enumerate(factors)]) if precondition else beta
        result=minimize(transformed,initial,jac=True,method='L-BFGS-B',
             options=dict(maxiter=maxiter,ftol=1e-14 if strict else 1e-10,gtol=1e-10 if strict else 1e-7,maxcor=15,maxls=100))
        result.x=to_beta(result.x)
        if strict:result.x=self.strict_polish(result.x)
        result.fun,result.jac=self.objective(result.x)
        # Near a stationary warm start, a line search can exhaust machine
        # precision without satisfying its relative function-change check.
        # Accept only the explicitly tighter normalized raw-gradient criterion.
        if not result.success and np.isfinite(result.fun) and np.max(np.abs(result.jac))<=1e-8:
            result.success=True;result.message='stationary normalized gradient <=1e-8 after line-search stop'
        if strict:result.success=bool(np.max(np.abs(result.jac))<1e-8)
        return result
    def heldout_loss(self,beta):
        sums=np.zeros(2);weights=np.zeros(2);rows=self.prepared.rows
        for sl in rows.slices(self.chunk):
            z=self.logits(beta,sl);held=~self.prepared.train[sl]
            for y in (0,1):
                mask=held&(rows.y[sl]==y);w=rows.w[sl][mask]
                sums[y]+=float(w@(np.logaddexp(0,z[mask])-y*z[mask]));weights[y]+=w.sum()
        return float(np.sum(sums/(2*weights))) if np.min(weights)>0 else None
    def surface(self,beta):
        prep=self.prepared;rows=prep.rows;P=self.p
        cr,cx=unpack(beta[:P],self.basis.sizes,self.basis.H)
        surface=dict(root_key=ast.literal_eval(rows.meta['root']),feature_dimension=rows.dim,
            coef_r=cr,coef_rx=cx,basis_specs=self.basis.spec,radial_window=rows.meta['radial_window'],
            support_basis_spec=prep.support_spec,**prep.support,context_corrections={})
        for b,c in enumerate(prep.corrections,1):
            cr,cx=unpack(beta[b*P:(b+1)*P],self.basis.sizes,self.basis.H)
            surface['context_corrections'][rows.dataset['contexts'][c['context_id']]]=dict(
                coef_r=cr,coef_rx=cx,support_basis_spec=prep.support_spec,**c['support'])
        return surface


def _cv_fold_worker(directory,root_id,spec,held,mass,fold,strict,interior_knots=(4,6)):
    """Run one independent fold; its optimizer and numerical libraries stay single-threaded."""
    import resource
    rows=RootRows(directory,root_id)
    grid=[dict(interior_knots=n,lambda_r=r,lambda_z=z,lambda_0=a) for n,r,z,a in
          product(interior_knots,(.01,.1,1.,10.),(.01,.1,1.,10.),(.001,.01,.1))]
    result_losses={tuple(c.values()):[] for c in grid};details=[]
    prep=PreparedRows(rows,spec,held,mass)
    if np.min(prep.root_counts)<20:
        for cfg in grid:result_losses[tuple(cfg.values())].append(float(np.log(2.)))
        details.append(dict(fold=fold,status='root_unsupported'))
    else:
        for n in interior_knots:
            path=sorted([c for c in grid if c['interior_knots']==n],key=lambda c:(-c['lambda_r'],-c['lambda_z'],-c['lambda_0']))
            start=None;cache={}
            for cfg in path:
                key=(cfg['lambda_r'],cfg['lambda_0'],cfg['lambda_z'] if prep.corrections else None)
                if key not in cache:
                    problem=ChunkedProblem(prep,cfg);fit=problem.fit(start,strict=strict)
                    if not fit.success:raise RuntimeError(f'nonconverged CV fit fold={fold} config={cfg} max_gradient={np.max(np.abs(fit.jac))}: {fit.message}')
                    start=fit.x;cache[key]=(problem.heldout_loss(fit.x),fit.nit)
                loss,nit=cache[key];result_losses[tuple(cfg.values())].append(loss)
            details.append(dict(fold=fold,knots=n,fits=len(cache),warm_start=True))
            print(f'CV fold {fold+1}/5 knots={n} completed {len(cache)} fits',flush=True)
    return dict(losses=result_losses,details=details,pid=os.getpid(),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)


def grouped_cv_root(rows,spec,strict=False,mass_cache=None,workers=1,folds=None,interior_knots=(4,6)):
    """Grouped CV grid; optional process parallelism is only across independent folds."""
    mass=mass_cache or support_mass(rows,spec);groups=mass[-1]
    folds=folds if folds is not None else grouped_folds(groups)
    if len(folds)!=5 or set(g for fold in folds for g in fold)!=set(groups):
        raise ValueError('fold assignment must partition the root groups into five folds')
    grid=[dict(interior_knots=n,lambda_r=r,lambda_z=z,lambda_0=a) for n,r,z,a in
          product(interior_knots,(.01,.1,1.,10.),(.01,.1,1.,10.),(.001,.01,.1))]
    losses={tuple(c.values()):[] for c in grid};details=[]
    if workers<1 or workers>4:raise ValueError('workers must be between 1 and 4')
    if workers==1:
        fold_results=[_cv_fold_worker(rows.directory,rows.meta['id'],spec,held,mass,fold,strict,interior_knots)
                      for fold,held in enumerate(folds)]
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as executor:
            futures=[executor.submit(_cv_fold_worker,rows.directory,rows.meta['id'],spec,held,mass,fold,strict,interior_knots)
                     for fold,held in enumerate(folds)]
            fold_results=[future.result() for future in as_completed(futures)]
    fold_results.sort(key=lambda item:item['details'][0]['fold'])
    worker_peaks=[]
    for item in fold_results:
        worker_peaks.append(dict(pid=item['pid'],peak_rss_mib=item['peak_rss_mib']));details.extend(item['details'])
        for key,values in item['losses'].items():losses[key].extend(values)
    report=[]
    for cfg in grid:
        ll=np.array(losses[tuple(cfg.values())]);report.append(dict(config=cfg,fold_losses=ll.tolist(),mean=float(ll.mean()),se=float(ll.std(ddof=1)/np.sqrt(5))))
    best=min(report,key=lambda e:e['mean']);eligible=[r for r in report if r['mean']<=best['mean']+best['se']]
    chosen=min(eligible,key=lambda r:(r['config']['interior_knots'],-r['config']['lambda_z'],-r['config']['lambda_0'],-r['config']['lambda_r']))
    prep=PreparedRows(rows,spec,mass_cache=mass);problem=ChunkedProblem(prep,chosen['config']);result=problem.fit(strict=strict)
    if not result.success:raise RuntimeError(f'nonconverged final fit: {result.message}')
    return problem.surface(result.x),dict(candidate_count=len(grid),folds=folds,results=report,best=best,selected=chosen,execution=details,
        cv_workers=workers,worker_peak_rss_mib=worker_peaks)
