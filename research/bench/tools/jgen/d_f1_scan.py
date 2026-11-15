"""Rigid crystal-frame Q1/Q2/S_orient scans for the registered D-B F1 keys."""
from __future__ import annotations
import json, math, sys, traceback, io, logging
from pathlib import Path
import numpy as np
import torch

from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from tools.jdev import multiseed
from jacobik.core.env import build_builder_and_inputs
from tools.runkit import mace

OUT=Path(__import__('os').environ.get('D_F1_OUT','/content/out/d_f1' if Path('/content').exists() else
    '/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/D/f1'))
MAP_DIR=Path('/content/out/d_f1_maps') if Path('/content').exists() else Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A4b/xyz')
KBT=2.479

def rotate(X, indices, origin, axis, deg):
    axis=np.asarray(axis,float); axis/=max(np.linalg.norm(axis),1e-12)
    a=math.radians(float(deg)); c,s=math.cos(a),math.sin(a)
    R=c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    Y=X.copy(); ids=np.asarray(indices,int); Y[ids]=(X[ids]-origin)@R.T+origin
    return Y

def unit(v):
    v=np.asarray(v,float); return v/max(float(np.linalg.norm(v)),1e-12)

def main(codes):
    torch.set_num_threads(1)
    stats=stats_from_db(inputs.DB_PATH); ev=mace.Evaluator(device='cuda',dtype='float64',threads=1)
    shards=inputs.frame_shards(); OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for code in codes:
      rec={'code':code}
      try:
        shard,dataset=shards[code]; input_mol=inputs.load_mol(code,shard,dataset)
        builder,_,err=build_builder_and_inputs(input_mol)
        if builder is None: raise RuntimeError(f'builder failed: {err}')
        logstream=io.StringIO(); handler=logging.StreamHandler(logstream); logger=logging.getLogger('jacobik.builder.engine'); logger.addHandler(handler)
        try: ctx,assign,bw,_=multiseed.prepare(builder,11,device='cpu',angle_bends=True,stats=stats)
        finally: logger.removeHandler(handler)
        if 'GFN2 embed refinement failed' in logstream.getvalue():
            raise RuntimeError('GFN2 embed refinement failed; refusing an MMFF fallback shell')
        (fks,_,ge,sc),seedrows=multiseed.seed_rows(builder,ctx,assign,bw,(11,))
        mol=ctx.mol
        mp=MAP_DIR/code/f'{code}_control_s11_pool.json' if MAP_DIR.name=='xyz' else MAP_DIR/f'{code}.json'
        manifest=json.loads(mp.read_text())
        ref_symbols=list(manifest['ref_symbols']); ref_to_gen=np.asarray(manifest['gen_to_ref_idx'],int)
        ref_X=np.asarray(manifest['ref_X'],float)
        if len(ref_to_gen)!=len(mol.atoms) or sorted(ref_to_gen.tolist())!=list(range(len(ref_to_gen))):
            raise ValueError('A4b reference-to-generator atom map does not match the scorer molecule')
        gen_symbols=[a.symbol for a in mol.atoms]
        if [gen_symbols[j] for j in ref_to_gen] != ref_symbols:
            raise ValueError('A4b generator/reference symbol mapping does not match the scorer molecule')
        gen_to_ref=np.argsort(ref_to_gen)
        X0=ref_X[gen_to_ref]
        pos={a.idx:i for i,a in enumerate(mol.atoms)}; m=pos[mol.metal_idx]
        # Ligands and kappa donor atoms eligible for rigid branch scans.
        ligand_of={idx:li for li,lig in enumerate(mol.ligands) for idx in lig.atom_indices}
        lig_atoms={li:[pos[x] for x in lig.atom_indices if x in pos] for li,lig in enumerate(mol.ligands)}
        donors=[]
        for site in mol.coord_sites:
            if len(site.donor_indices)==1:
                d=site.donor_indices[0]; li=ligand_of.get(d)
                if li is not None: donors.append({'d':d,'li':li,'indices':lig_atoms[li]})
        scans=[]
        # Q1: radial M-D translation of monodentate ligand branches.
        for d in donors:
            di=pos[d['d']]; v=unit(X0[di]-X0[m]); grid=sorted(set(np.round(np.r_[np.linspace(-.3,1.0,14),0.0],4)))
            structures=[]; coords=[]
            for dx in grid:
                X=X0.copy(); X[d['indices']]+=dx*v; coords.append(X)
                structures.append((ref_symbols,X[ref_to_gen],mol.charge,mol.spin_state))
            scans.append({'term':'Q1','donor':d['d'],'offsets':grid,'coords':coords})
        # Q2: open donor pairs by rigidly rotating one monodentate ligand around M.
        for i,a in enumerate(donors):
          for b in donors[i+1:]:
            if a['li']==b['li']: continue
            ia,ib=pos[a['d']],pos[b['d']]; va=unit(X0[ia]-X0[m]); vb=unit(X0[ib]-X0[m]); axis=np.cross(va,vb)
            if np.linalg.norm(axis)<1e-6: axis=np.cross(va,np.array([1.,0.,0.]))
            grid=sorted(set(np.r_[np.linspace(-40,40,9),0.0].tolist())); coords=[]
            for da in grid: coords.append(rotate(X0,b['indices'],X0[m],axis,da))
            scans.append({'term':'Q2','donors':[a['d'],b['d']],'offsets':grid,'coords':coords})
        # S_orient tilt: rotate monodentate ligand around a transverse axis through its donor.
        for d in donors:
            di=pos[d['d']]; md=unit(X0[di]-X0[m]); axis=np.cross(md,np.array([0.,0.,1.]))
            if np.linalg.norm(axis)<1e-6: axis=np.cross(md,np.array([0.,1.,0.]))
            grid=sorted(set(np.r_[np.linspace(-40,40,9),0.0].tolist())); coords=[]
            for angle in grid: coords.append(rotate(X0,d['indices'],X0[di],axis,angle))
            scans.append({'term':'S_orient','donor':d['d'],'offsets':grid,'coords':coords})
        flat=[(ref_symbols,X[ref_to_gen],mol.charge,mol.spin_state) for s in scans for X in s['coords']]
        E=(ev.energies_kj(flat)/KBT).tolist()
        cursor=0; scored=[]
        for scan in scans:
            n=len(scan['coords']); e=np.asarray(E[cursor:cursor+n]); cursor+=n
            Xt=torch.as_tensor(np.stack(scan['coords']),dtype=torch.float32)
            rg=torch.full((n,),int(np.asarray(assign).reshape(-1)[0]),dtype=torch.long)
            with torch.no_grad(): _,terms=sc.score(Xt,row_groups=rg)
            name={'Q1':'Q1','Q2':'Q2','S_orient':'Q3'}[scan['term']]
            profile=(-terms[name].detach().cpu().numpy()*float(sc._agg_norm())).tolist()
            e0=float(e[scan['offsets'].index(0.0)]); p0=float(profile[scan['offsets'].index(0.0)])
            scored.append({k:v for k,v in scan.items() if k!='coords'}|{'deltaE_kT':(e-e0).tolist(),'term_penalty_delta':(np.asarray(profile)-p0).tolist()})
        rec.update(status='ok',fingerprint=__import__('jacobik.core.profile',fromlist=['snapshot']).snapshot()['fingerprint'],
                   model_sha256=ev.model_sha256,scans=scored,mace_stats=ev.stats)
      except Exception as exc:
        rec.update(status='failed',reason=f'{type(exc).__name__}: {exc}'[:500],trace=traceback.format_exc()[-2000:])
      results.append(rec); (OUT/f'{code}.json').write_text(json.dumps(rec)+'\n')
      print(f'{code}: {rec["status"]} scans={len(rec.get("scans",[]))}',flush=True)
    print(json.dumps({'parents':len(results),'ok':sum(r.get('status')=='ok' for r in results),'failures':[r for r in results if r.get('status')!='ok']},indent=2))

if __name__=='__main__': main(sys.argv[1:] or ['QEZVEY','VAVBIE','USAPOU','AZUSIA'])
