"""VM-side GFN2 and CUDA MACE driver for A4b pool manifests."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from tools.jgen.bench.pool_export import _read_frames, reference_context
from tools.runkit import xtb, mace, geom
from .relax_set import select

def to_ref_order(X, gen_to_ref_idx):
    """Apply the bench's explicit builder-to-reference coordinate index sequence."""
    return np.asarray(X)[np.asarray(gen_to_ref_idx, dtype=int)]

#: per-row fields recomputed from the relaxed geometry on every run (never reused from a previous run)
DERIVED = ("hit", "equivalent", "rmsd_refrel_A", "offset", "contact_change", "status", "E_mace_kj", "X")

def load_prior(resume):
    """Previous run's relax dir -> ({(code, seed, row): row}, {code: reference}); empty when resume is None."""
    rows, refs = {}, {}
    if not resume or not Path(resume).is_dir():
        return rows, refs
    for f in Path(resume).glob("*_*.json"):
        if f.name in ("references.json", "failures.json", "run_metrics.json"):
            continue
        try:
            d = json.loads(f.read_text())
            for k, r in d.get("rows", {}).items():
                rows[(d["code"], int(d["seed"]), int(k))] = r
        except Exception:
            pass
    rp = Path(resume) / "references.json"
    if rp.exists():
        refs = json.loads(rp.read_text())
    return rows, refs

def _x0_sha(X):
    """Hash of an unrelaxed geometry (1e-4 A grid): the identity of a pool row for resume."""
    import hashlib
    return hashlib.sha256(np.round(np.asarray(X, float), 4).tobytes()).hexdigest()[:16]

def _a4c_selection(manifests, out, nproc):
    """A4c relax-set rule (a4c.relax_select) per manifest, in a process pool; log written to out/selection.json."""
    from concurrent.futures import ProcessPoolExecutor
    chosen, log = {}, {}
    if nproc > 1 and len(manifests) > 1:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            res = list(ex.map(_safe_select, manifests, chunksize=1))
    else:
        res = [_safe_select(mp) for mp in manifests]
    for mp, r, err in res:
        if r is None:
            log[str(mp)] = dict(error=err)
        else:
            chosen[str(mp)] = r[0]; log[str(mp)] = r[1]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "selection.json").write_text(json.dumps(log, default=int))
    return chosen, log

def _safe_select(mp):
    """A failed selection is recorded; the unit becomes a failed record in process()."""
    from tools.jgen.a4c.relax_select import select_manifest
    try:
        return mp, select_manifest(mp), None
    except Exception as e:
        return mp, None, f"{type(e).__name__}: {e}"

def process(manifests, out, charge_mult, nproc=4, evaluator=None, resume=None, selector="a4b"):
    """Process units independently; malformed units become failed JSON records.

    resume: a previous run's relax dir. Rows and references relaxed there (X present, MACE energy present when the
    relaxation was ok) are reused with their stored energies: xtb and MACE run only for the rest. Same inputs (frozen
    selection, same records) give the same rows, so a resumed run equals a fresh one up to the MACE pool order.
    selector: 'a4b' (frozen A4b rule: top-1, floor, <= 3 rows within 1.5 A full RMSD) or 'a4c' (class-aware rule,
    tools.jgen.a4c.relax_select; per-unit log in out/selection.json)."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    xtb.require(); pending=[]; refs={}; failures=[]
    if selector not in ("a4b", "a4c"): raise ValueError(f"unknown selector {selector!r}")
    a4c_rows, a4c_log = _a4c_selection(manifests, out, nproc) if selector == "a4c" else ({}, {})
    contexts={}
    for mp in manifests:
        try:
            mp=Path(mp); m=json.loads(mp.read_text()); recp=mp.parents[2]/"records"/f"{m['code']}_{m['seed']}.json"
            rec=json.loads(recp.read_text()); code=m['code']
            if selector=='a4c':
                if str(mp) not in a4c_rows: raise ValueError(f"a4c selection failed: {a4c_log.get(str(mp),{}).get('error')}")
                if int(m['n_rows'])!=int(rec['n_rows']): raise ValueError('pool/record row count mismatch')
                idx=a4c_rows[str(mp)]
            else: idx=select(m,rec)
            q,mult=int(charge_mult[code]['charge']),int(charge_mult[code]['mult'])
            frames=list(_read_frames(mp.parent/m['rows'][0]['file']))
            if len(frames)!=len(m['rows']): raise ValueError('XYZ/manifest row count mismatch')
            mapping=np.asarray(m['gen_to_ref_idx'],int); symbols=m['ref_symbols']
            if code not in refs:
                refs[code]=(symbols,np.asarray(m['ref_X'],float),q,mult)
                contexts[code]=reference_context(symbols,np.asarray(m['ref_X'],float))
            for i in idx:
                _,ss,xx=frames[i]; pending.append(dict(code=code,seed=m['seed'],row=i,symbols=symbols,X=to_ref_order(xx,mapping),q=q,mult=mult,manifest=str(mp)))
        except Exception as e:
            failures.append(dict(manifest=str(mp),status='failed',reason=f'{type(e).__name__}: {e}'))
    prior_rows, prior_refs = load_prior(resume)
    def _reusable(r):   # relaxed and measured; failed rows are retried
        return r is not None and r.get('status') in ('ok', 'capped') and r.get('X') is not None and r.get('E_mace_kj') is not None
    reuse = {i: prior_rows.get((p['code'], int(p['seed']), int(p['row']))) for i, p in enumerate(pending)}
    # a row is reused only when its unrelaxed geometry is the same (X0_sha; rows without it are never reused): another
    # arm or another solve of the same arm can put a different structure at the same (code, seed, row)
    reuse = {i: r for i, r in reuse.items() if _reusable(r) and r.get('X0_sha') == _x0_sha(pending[i]['X'])}
    ref_codes = list(refs)
    ref_reuse = {c: prior_refs[c] for c in ref_codes if c in prior_refs and prior_refs[c].get('X') is not None
                 and prior_refs[c].get('E_mace_kj') is not None and prior_refs[c].get('E_unrel_mace_kj') is not None}
    new_rows = [i for i in range(len(pending)) if i not in reuse]
    new_refs = [k for k, c in enumerate(ref_codes) if c not in ref_reuse]
    # Keep each xtb invocation independent, then submit larger systems first to the pool.
    jobs=[(pending[i]['symbols'],pending[i]['X'],pending[i]['q'],pending[i]['mult']) for i in new_rows]
    refjobs_all=list(refs.values())
    refjobs=[refjobs_all[k] for k in new_refs]
    # one pool for pool rows and references (independent jobs, same results), largest first: one tail, not two
    alljobs=jobs+refjobs
    order=sorted(range(len(alljobs)),key=lambda i:-len(alljobs[i][0]))
    rr=xtb.opt_many([alljobs[i] for i in order],workers=max(1,nproc-1),workroot=str(out/'tmp'),cycles=200) if alljobs else []
    allres=[None]*len(alljobs)
    for i,r in zip(order,rr): allres[i]=r
    relax=[None]*len(pending); refres=[None]*len(ref_codes)
    for i,r in zip(new_rows,allres[:len(jobs)]): relax[i]=r
    for k,r in zip(new_refs,allres[len(jobs):]): refres[k]=r
    energy_map={}
    for i,r in reuse.items():   # xtb-level result: status 'ok' (capped rows converged=False keep cycles)
        relax[i]=dict(**{k:v for k,v in r.items() if k not in DERIVED},status='ok',X=np.asarray(r['X'],float))
        if r.get('E_mace_kj') is not None: energy_map[('row',i)]=float(r['E_mace_kj'])
    for k,c in enumerate(ref_codes):
        if c in ref_reuse:
            r=ref_reuse[c]
            refres[k]=dict(**{kk:v for kk,v in r.items() if kk not in ('X','E_mace_kj','E_unrel_mace_kj')},X=np.asarray(r['X'],float))
            energy_map[('ref',k)]=float(r['E_mace_kj']); energy_map[('ref_unrel',k)]=float(r['E_unrel_mace_kj'])
    ev=evaluator if evaluator is not None else mace.Evaluator(device='cuda'); structures=[]; slots=[]
    for k,(s,x,q,m) in enumerate(refjobs_all):
        if ref_codes[k] in ref_reuse: continue
        r=refres[k]
        slots.append(('ref_unrel',k)); structures.append((s,x,q,m))
        if r.get('X') is not None and r.get('status')=='ok':
            slots.append(('ref',k)); structures.append((s,r['X'],q,m))
    for i in new_rows:
        z,p=relax[i],pending[i]
        if z.get('X') is not None and z.get('status')=='ok':
            slots.append(('row',i)); structures.append((p['symbols'],z['X'],p['q'],p['mult']))
    energies=ev.energies_kj(structures) if structures else np.zeros(0)
    energy_map.update({k:float(v) for k,v in zip(slots,energies)})
    refout={}
    for i,(code,r) in enumerate(zip(refs,refres)):
        refout[code]=dict(**{k:v for k,v in r.items() if k!='X'},X=None if r.get('X') is None else r['X'].tolist(),E_mace_kj=energy_map.get(('ref',i)),E_unrel_mace_kj=energy_map.get(('ref_unrel',i)))
    (out/'references.json').write_text(json.dumps(refout,allow_nan=False))
    for i,(p,r) in enumerate(zip(pending,relax)):
        status=('capped' if r.get('cycles') is not None and int(r['cycles'])>=200 else ('ok' if r.get('status')=='ok' else 'failed'))
        e=energy_map.get(('row',i)); r=dict(**{k:v for k,v in r.items() if k not in ('X','status','E_mace_kj')},status=status,X=None if r.get('X') is None else r['X'].tolist(),E_mace_kj=e)
        r['X0_sha']=_x0_sha(p['X'])
        if e is None: r.update(hit=False,equivalent=False,contact_change=None)
        else:
            ref=refout[p['code']]; Xr=np.asarray(r['X']); Xref=np.asarray(ref['X']) if ref.get('X') is not None else None
            rmsd=float(geom.sym_rmsd_to_ref(Xr[np.array([s!='H' for s in p['symbols']])][None,:,:],Xref[np.array([s!='H' for s in p['symbols']])],contexts[p['code']][1],graph=contexts[p['code']][3] if len(contexts[p['code']])>3 else None)[0]) if Xref is not None else float('inf')
            off=e-ref['E_mace_kj'] if ref.get('E_mace_kj') is not None else float('inf')
            r.update(rmsd_refrel_A=(rmsd if np.isfinite(rmsd) else None),offset=(off if np.isfinite(off) else None),hit=bool(rmsd<=.3),equivalent=bool(rmsd>.3 and off<=5))
            metal=geom.find_metal(p['symbols']); rc=np.array([geom.rcov(s) for s in p['symbols']]); heavy=np.array([s!='H' for s in p['symbols']]);
            def contacts(X): return sorted(int(j) for j in np.where(heavy)[0] if j!=metal and np.linalg.norm(X[j]-X[metal])<=rc[metal]+rc[j]+.6)
            r['contact_change']=contacts(np.asarray(p['X']))!=contacts(Xr)
        path=out/f"{p['code']}_{p['seed']}.json"; data=json.loads(path.read_text()) if path.exists() else dict(code=p['code'],seed=p['seed'],rows={})
        data['rows'][str(p['row'])]=r; path.write_text(json.dumps(data,allow_nan=False))
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    metrics=dict(n_relax=len(pending),n_reference=len(refjobs_all),n_failed=len(failures),
                 n_reused_rows=len(reuse),n_reused_refs=len(ref_reuse),
                 xtb_wall_sum_s=float(sum(float(r.get('wall_s') or 0) for r in allres)),
                 xtb_wall_max_s=float(max([float(r.get('wall_s') or 0) for r in allres] or [0])),mace=ev.stats)
    (out/'run_metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifests',nargs='+'); ap.add_argument('--out',required=True); ap.add_argument('--charge-mult',required=True); ap.add_argument('--nproc',type=int,default=4); a=ap.parse_args()
    print(json.dumps(process(a.manifests,a.out,json.loads(Path(a.charge_mult).read_text()),a.nproc)))
if __name__=='__main__': main()
