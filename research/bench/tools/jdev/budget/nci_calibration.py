"""Independent physical NCI probes: frozen grids, strict ligand-only GFN2,
MACE64 interaction energies, complete Q6 replacement and per-contact scores.
No runtime defaults, statistical fits, solver steering or benchmark inputs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE

OUT=REPO/'data/refit/nci/2026-10-18_energy_calibration_v2'
MONOMERS={'water':'O','aniline':'Nc1ccccc1','pyridine':'c1ccncc1','acetone':'CC(C)=O',
          'sulfonate':'CS(=O)(=O)[O-]','chloride':'[Cl-]','benzene':'c1ccccc1','methane':'C','ethane':'CC'}
HB=[('water','water'),('water','pyridine'),('water','acetone'),('aniline','water'),('aniline','pyridine'),
    ('water','sulfonate'),('aniline','sulfonate'),('water','chloride'),('aniline','chloride')]
CONFIG=dict(version=1,monomers=MONOMERS,hbond_pairs=HB,
    hb_r=[float(round(x,2)) for x in np.arange(1.6,3.61,.2)]+[6.],hb_angle=[90.,120.,150.,180.],
    pp_pairs=[('benzene','benzene'),('benzene','pyridine')],pp_height=[float(round(x,2)) for x in np.arange(3.,5.51,.25)]+[8.],
    pp_slip=[0.,1.5,3.],pp_angle=[0.,45.,90.],chp_pairs=[('methane','benzene'),('ethane','benzene')],
    chp_r=[float(round(x,2)) for x in np.arange(2.,4.01,.2)]+[7.],chp_direction=[60.,90.,120.,150.,180.],chp_approach=[0.,45.,75.],
    units=dict(energy='kJ/mol',distance='Angstrom',angle='degree',score='raw nats'),
    primary='MACE float64 interaction energy; GFN2 independent crosscheck at identical rigid geometries',
    hypothesis='Current unit-weight NCI fails at least one H-bond replacement/transfer category; neutral pi signal is expected but not assumed transferable.',
    acceptance=dict(curvature_ratio=[.7,1.5],no_validation_wrong_sign_increase=True,no_dev_fitting=True),
    constraints='No fitted candidate until independent training validation. No global term reweighting, default change, complex optimization, held-out data or added force field.')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    from tools.jdev.env import tree_hash
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads(json.dumps(CONFIG))
    manifest=OUT/'manifest.json'
    value=dict(config=config,source_hash=tree_hash(),db_sha256=digest(CANDIDATE),model_sha256=digest(REPO/'tools/jdev/MACE-POLAR-1-S.model'),
               driver_sha256=digest(Path(__file__)))
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old==value,'calibration provenance changed; use a new output directory'
    else:write(manifest,value)
    return value


def frame(axis):
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    v=np.eye(3)[np.argmin(abs(axis))];v-=axis*(v@axis);v/=np.linalg.norm(v)
    return np.column_stack([axis,v,np.cross(axis,v)])


def orient(x,origin,axis):
    return (x-origin)@frame(axis)


def rotate_y(angle):
    a=np.radians(angle);return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])


def xtb(rd,x,opt=False):
    from rdkit import Chem
    from jacobik.builder.engine import XTB_BIN
    symbols=[a.GetSymbol() for a in rd.GetAtoms()];q=Chem.GetFormalCharge(rd)
    mult=(sum(a.GetAtomicNum() for a in rd.GetAtoms())-q)%2+1
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);(td/'m.xyz').write_text(f'{len(symbols)}\n\n'+''.join(f'{s} {a:.9f} {b:.9f} {c:.9f}\n' for s,(a,b,c) in zip(symbols,x)))
        args=[XTB_BIN,'m.xyz','--gfn','2','--chrg',str(q),'--uhf',str(mult-1)]
        args+=['--opt','tight'] if opt and len(symbols)>1 else ['--sp']
        p=subprocess.run(args,cwd=td,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180,
                         env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        values=re.findall(rb'TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh',p.stdout)
        if p.returncode or not values:raise RuntimeError(f'GFN2 exit {p.returncode}: '+p.stdout[-700:].decode(errors='replace'))
        y=np.array(x,copy=True)
        if opt and len(symbols)>1:
            y=np.array([[float(v) for v in line.split()[1:4]] for line in (td/'xtbopt.xyz').read_text().splitlines()[2:2+len(symbols)]])
        assert y.shape==x.shape and np.isfinite(y).all()
        return float(values[-1])*2625.4996394799,y


def molecules():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    out={}
    for name,smi in MONOMERS.items():
        rd=Chem.AddHs(Chem.MolFromSmiles(smi));path=OUT/f'monomer_{name}.json'
        if path.exists():obj=json.loads(path.read_text());x=np.array(obj['X'])
        else:
            if rd.GetNumAtoms()>1:
                assert AllChem.EmbedMolecule(rd,randomSeed=42)>=0
                if AllChem.MMFFHasAllMoleculeParams(rd):AllChem.MMFFOptimizeMolecule(rd,maxIters=2000)
                x=rd.GetConformer().GetPositions()
            else:x=np.zeros((1,3))
            energy,x=xtb(rd,x,opt=True)
            obj=dict(smiles=smi,symbols=[a.GetSymbol() for a in rd.GetAtoms()],X=x.tolist(),gfn2=energy,charge=Chem.GetFormalCharge(rd),mult=1)
            write(path,obj)
        out[name]=(rd,x,obj)
    return out


def topology(rdA,rdB):
    import torch
    from jacobik.core.types import Atom,Bond
    from jacobik.stats import keys as kb
    atoms=[];bonds=[];offset=0
    for rd in (rdA,rdB):
        atoms.extend(Atom(offset+a.GetIdx()+1,a.GetSymbol(),np.zeros(3)) for a in rd.GetAtoms())
        bonds.extend(Bond(offset+b.GetBeginAtomIdx()+1,offset+b.GetEndAtomIdx()+1,b.GetBondTypeAsDouble()) for b in rd.GetBonds())
        offset+=rd.GetNumAtoms()
    mol=SimpleNamespace(atoms=atoms,bonds=bonds,metal_idx=-1,coord_sites=[])
    polar=kb.q6_polar_h(mol)
    pairs=[(a,b) for a in range(rdA.GetNumAtoms()) for b in range(rdA.GetNumAtoms(),len(atoms))]
    ge=SimpleNamespace(_q6_a=torch.tensor([a for a,b in pairs]),_q6_b=torch.tensor([b for a,b in pairs]),
        _q6_elem_keys=[kb.q6_pair_key(atoms[a].symbol,atoms[b].symbol,a+1 in polar,b+1 in polar) for a,b in pairs])
    return mol,ge


def score_grid(rdA,rdB,coords,stats):
    import torch
    from research.archive.code.interactions import nci,q6pmf
    mol,ge=topology(rdA,rdB);n=len(ge._q6_a)
    qr=q6pmf.build_runtime(mol,ge,stats.q6_pmf_tables(),torch.full((n,),2.),torch.full((n,),.3),'cpu')
    assert qr is not None
    rt=nci.build_runtime(mol,stats.nci_tables(),'cpu',types=nci.TYPES) or {}
    x=torch.tensor(np.array(coords),dtype=torch.float32)
    flag,types=nci.NCI_ON,nci.ACTIVE_TYPES
    try:
        nci.NCI_ON=False
        off=q6pmf.evaluate(qr,x,ge._q6_a,ge._q6_b).numpy()*q6pmf.RT_KJ
        nci.NCI_ON=True;nci.ACTIVE_TYPES=nci.TYPES
        core=q6pmf.evaluate(qr,x,ge._q6_a,ge._q6_b).numpy()*q6pmf.RT_KJ
    finally:nci.NCI_ON=flag;nci.ACTIVE_TYPES=types
    scores={t:nci.evaluate({t:e},x).numpy() for t,e in rt.items()}
    contacts={}
    for t,e in rt.items():
        r,a=nci.features(t,x,e['idx']);v=nci._bilinear(e['T'],e['row'],r,a,e['grid']).numpy()
        contacts[t]=dict(r=r.numpy().tolist(),angle=a.numpy().tolist(),score=v.tolist())
    return dict(q6_old=off,q6_core=core,**{'score_'+t:v for t,v in scores.items()}),dict(contacts=contacts,
        q6_pairs=n,q6_pairs_with_table=qr['n_pairs_with_table'],candidates={t:e['n'] for t,e in rt.items()})


def prepare():
    from rdkit import Chem
    import torch
    torch.set_num_threads(1);freeze();mons=molecules();stats=stats_from_db(CANDIDATE)
    for kind,pairs in [('hb',HB),('pp',CONFIG['pp_pairs']),('chp',CONFIG['chp_pairs'])]:
        for nameA,nameB in pairs:
            stem=OUT/f'{kind}_{nameA}_{nameB}'
            if stem.with_suffix('.npz').exists():continue
            rdA,A,metaA=mons[nameA];rdB,B,metaB=mons[nameB];coords=[];labels=[]
            if kind=='hb':
                d=next(a.GetIdx() for a in rdA.GetAtoms() if a.GetSymbol() in ('O','N') and any(n.GetAtomicNum()==1 for n in a.GetNeighbors()))
                h=next(a.GetIdx() for a in rdA.GetAtomWithIdx(d).GetNeighbors() if a.GetAtomicNum()==1)
                aa=next(a.GetIdx() for a in rdB.GetAtoms() if a.GetSymbol() in ('O','N','Cl'))
                if nameB=='sulfonate':aa=next(a.GetIdx() for a in rdB.GetAtoms() if a.GetFormalCharge()<0)
                a=orient(A,A[h],A[h]-A[d])
                nb=[v.GetIdx() for v in rdB.GetAtomWithIdx(aa).GetNeighbors()]
                axis=(B[nb]-B[aa]).sum(0) if nb else np.array([1.,0,0])
                b=orient(B,B[aa],axis)
                for angle in CONFIG['hb_angle']:
                    phi=np.radians(180-angle);u=np.array([np.cos(phi),np.sin(phi),0.]);bb=b@frame(u).T
                    for r in CONFIG['hb_r']:
                        coords.append(np.vstack([a,bb+r*u]));labels.append(dict(scan=f'a{angle}',r=r,angle=angle,far=r==6.))
            elif kind=='pp':
                def ringplane(x,rd):
                    rows=[a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum()>1];center=x[rows].mean(0)
                    _,_,vt=np.linalg.svd(x[rows]-center);vt[-1]*=np.linalg.det(vt)
                    return (x-center)@vt.T
                a,b=ringplane(A,rdA),ringplane(B,rdB)
                for angle in CONFIG['pp_angle']:
                    bb=b@rotate_y(angle).T
                    for slip in CONFIG['pp_slip']:
                        for r in CONFIG['pp_height']:
                            coords.append(np.vstack([a,bb+[slip,0,r]]));labels.append(dict(scan=f'a{angle}_s{slip}',r=r,angle=angle,slip=slip,far=r==8.))
            else:
                c=0;h=next(a.GetIdx() for a in rdA.GetAtomWithIdx(c).GetNeighbors() if a.GetAtomicNum()==1)
                a=orient(A,A[h],A[c]-A[h]);heavy=[a.GetIdx() for a in rdB.GetAtoms() if a.GetAtomicNum()>1]
                center=B[heavy].mean(0);_,_,vt=np.linalg.svd(B[heavy]-center);vt[-1]*=np.linalg.det(vt);b=(B-center)@vt.T
                for approach in CONFIG['chp_approach']:
                    phi=np.radians(approach);u=np.array([np.sin(phi),0,np.cos(phi)]);v=np.array([np.cos(phi),0,-np.sin(phi)])
                    for direction in CONFIG['chp_direction']:
                        theta=np.radians(direction);aa=a@frame(np.cos(theta)*u+np.sin(theta)*v).T
                        for r in CONFIG['chp_r']:
                            coords.append(np.vstack([aa,b+r*u]));labels.append(dict(scan=f'p{approach}_a{direction}',r=r,angle=direction,approach=approach,far=r==7.))
            values,extra=score_grid(rdA,rdB,coords,stats)
            np.savez_compressed(stem.with_suffix('.npz'),X=np.array(coords),**values)
            write(stem.with_suffix('.json'),dict(type=kind,a=nameA,b=nameB,labels=labels,
                symbols=metaA['symbols']+metaB['symbols'],charge=metaA['charge']+metaB['charge'],mult=1,**extra))
            print(stem.name,len(coords),'prepared',flush=True)


def measure():
    import torch
    from rdkit import Chem
    from jacobik.core.energy import MACEPolarEvaluator
    freeze();torch.set_num_threads(1);mons=molecules()
    ev=MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',threads=1,model_path=REPO/'tools/jdev/MACE-POLAR-1-S.model')
    monE=ev.energies([(m['symbols'],x,m['charge'],m['mult']) for rd,x,m in mons.values()])*96.4853321233
    monE=dict(zip(mons,map(float,monE)));write(OUT/'monomer_mace.json',monE)
    for f in sorted(OUT.glob('hb_*.npz'))+sorted(OUT.glob('pp_*.npz'))+sorted(OUT.glob('chp_*.npz')):
        meta=json.loads(f.with_suffix('.json').read_text());target=f.with_suffix('.mace64.json')
        if target.exists():continue
        with np.load(f) as z:x=z['X']
        energies=ev.energies([(meta['symbols'],y,meta['charge'],meta['mult']) for y in x])*96.4853321233-monE[meta['a']]-monE[meta['b']]
        assert np.isfinite(energies).all()
        write(target,dict(interaction=energies.tolist()));print(f.stem,'MACE64',len(energies),flush=True)
    del ev;torch.cuda.empty_cache()
    for f in sorted(OUT.glob('hb_*.npz'))+sorted(OUT.glob('pp_*.npz'))+sorted(OUT.glob('chp_*.npz')):
        target=f.with_suffix('.gfn2.json')
        if target.exists():continue
        meta=json.loads(f.with_suffix('.json').read_text());rd=Chem.CombineMols(mons[meta['a']][0],mons[meta['b']][0])
        with np.load(f) as z:coords=z['X']
        with ThreadPoolExecutor(max_workers=2) as executor:
            energies=list(executor.map(lambda x:xtb(rd,x)[0]-mons[meta['a']][2]['gfn2']-mons[meta['b']][2]['gfn2'],coords))
        write(target,dict(interaction=energies));print(f.stem,'GFN2',len(energies),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure']);a=p.parse_args()
    setup();prepare() if a.stage=='prepare' else measure()
