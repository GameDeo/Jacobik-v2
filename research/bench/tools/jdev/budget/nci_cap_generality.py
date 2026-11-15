"""Independent training-ligand countercheck of broad H-cap retention.

Selection is by chemistry and stable hash before evaluating any geometry error.
No benchmark/held-out inputs, complex relaxation, score fitting or defaults.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup
from energy_viewer import align_transform

OUT=REPO/'data/refit/reports/n2_cap_generality_2026-10-18'


def select():
    from rdkit import Chem
    from jacobik.builder.embed import cap_anions
    source=REPO/'data/refit/reports/oxidation_training_modes_2026-10-18.json'
    data=json.loads(source.read_text());groups={};seen=set();counts={}
    for row in data['rows']:
        for li in row['ligands']:
            smi=li['smiles'];donors=li['donors'];key=(smi,tuple(map(tuple,donors)))
            if key in seen or '[H]' in smi:continue
            seen.add(key);base=Chem.MolFromSmiles(smi)
            if base is None or Chem.AddHs(base).GetNumAtoms()>80:continue
            capped,caps=cap_anions(base,donors)
            if not caps or any(capped.GetAtomWithIdx(i).GetAtomicNum()!=1 for i in caps):continue
            parents=[capped.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in caps]
            labels=[]
            for p in parents:
                a=base.GetAtomWithIdx(p)
                label=('anion_' if a.GetFormalCharge()<0 else 'radical_')+a.GetSymbol()+('_aromatic' if a.GetIsAromatic() else '')
                if a.GetFormalCharge()<0 and Chem.GetFormalCharge(base)>=0:label='zwitterion_'+a.GetSymbol()
                labels.append(label)
            family='+'.join(sorted(set(labels)));counts[family]=counts.get(family,0)+1
            if any(base.GetAtomWithIdx(a).GetSymbol()=='N' and base.GetAtomWithIdx(b).GetSymbol()=='N'
                   and base.GetAtomWithIdx(a).GetFormalCharge()<0 and base.GetAtomWithIdx(b).GetFormalCharge()<0
                   and base.GetBondBetweenAtoms(a,b) is not None for j,a in enumerate(parents) for b in parents[j+1:]):
                family='adjacent_anionic_NN'
            item=dict(code=row['code'],shard=row['set'],smiles=smi,donors=donors,family=family,
                      cap_parents=parents,ligand_key=repr(key))
            key_family=family if family in ('anion_N','anion_O','anion_S','anion_C','anion_C_aromatic','anion_N_aromatic','radical_C','zwitterion_C','adjacent_anionic_NN') else ('mixed' if '+' in family else None)
            if key_family:groups.setdefault(key_family,[]).append(item)
    chosen=[]
    for family,items in sorted(groups.items()):
        chosen.extend(sorted(items,key=lambda x:zlib.crc32(('cap-generality-2026-10-18:'+x['ligand_key']).encode()))[:2])
    OUT.mkdir(exist_ok=True)
    selection=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),family_available=counts,selected=chosen,
                   selection='Two unique ligands per predeclared main-group H-only-cap family (N/O/S/C anions, aromatic C/N anions, radical C, zwitterionic C, adjacent anionic NN, and mixed), stable hash, <=80 atoms with H; from existing 7,353 mapped training structures with canonical benchmark exclusions. Maximum 20 ligands. Selection precedes geometry measurement; positive screen cannot establish generality.')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,selected=chosen),indent=2),flush=True)


def refine(rd,x,pairs):
    """Strict ligand-only xTB optimization with explicit donor-distance constraints."""
    from rdkit import Chem
    from jacobik.builder.engine import XTB_BIN
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);sy=[a.GetSymbol() for a in rd.GetAtoms()]
        (td/'l.xyz').write_text(f'{len(sy)}\n\n'+''.join(f'{s} {a:.9f} {b:.9f} {c:.9f}\n' for s,(a,b,c) in zip(sy,x)))
        q=Chem.GetFormalCharge(rd);uhf=(sum(a.GetAtomicNum() for a in rd.GetAtoms())-q)%2
        args=[XTB_BIN,'l.xyz','--gfn','2','--opt','crude','--chrg',str(q),'--uhf',str(uhf)]
        if pairs:
            (td/'c.inp').write_text('$constrain\n force constant=1.0\n'+''.join(f' distance: {a+1}, {b+1}, auto\n' for a,b in pairs)+'$end\n')
            args+=['--input','c.inp']
        result=subprocess.run(args,cwd=td,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'),timeout=180)
        if result.returncode or not (td/'xtbopt.xyz').exists():raise RuntimeError(f'xTB exit {result.returncode}: '+result.stdout[-600:].decode(errors='replace'))
        y=np.array([[float(v) for v in line.split()[1:4]] for line in (td/'xtbopt.xyz').read_text().splitlines()[2:2+len(sy)]])
        assert y.shape==x.shape and np.isfinite(y).all()
        return y


def cap_start(base,original,positions,donors):
    """Same original coordinates in both arms; append only temporary cap H."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from jacobik.builder.embed import cap_anions
    cap,caps=cap_anions(base,donors);sur=Chem.AddHs(cap)
    assert caps and all(sur.GetAtomWithIdx(i).GetAtomicNum()==1 for i in caps)
    p=AllChem.ETKDGv3();p.randomSeed=42
    assert AllChem.EmbedMolecule(sur,p)>=0
    AllChem.MMFFOptimizeMolecule(sur,maxIters=2000)
    y=sur.GetConformer().GetPositions();n=base.GetNumAtoms()
    R,t=align_transform(y[:n],positions[:n],range(n));y=y@R+t
    cap_vectors={i:y[i]-y[sur.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()] for i in caps}
    mapping=list(range(n));available={}
    for atom in sur.GetAtoms():
        if atom.GetAtomicNum()==1 and atom.GetIdx() not in caps:
            available.setdefault(atom.GetNeighbors()[0].GetIdx(),[]).append(atom.GetIdx())
    for atom in original.GetAtoms():
        if atom.GetIdx()<n:continue
        parent=atom.GetNeighbors()[0].GetIdx();mapping.append(available[parent].pop(0))
    y[mapping]=positions
    for i,v in cap_vectors.items():
        parent=sur.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx();y[i]=positions[parent]+v
    np.testing.assert_allclose(y[mapping],positions,atol=1e-12)
    return sur,y,mapping


def errors(rd,x,reference,donors):
    import itertools
    bonds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in rd.GetBonds() if b.GetBeginAtom().GetAtomicNum()>1 and b.GetEndAtom().GetAtomicNum()>1]
    bdiff=[np.linalg.norm(x[a]-x[b])-np.linalg.norm(reference[a]-reference[b]) for a,b in bonds]
    angle_errors=[]
    for a in sorted(set(i for g in donors for i in g)):
        nb=[n.GetIdx() for n in rd.GetAtomWithIdx(a).GetNeighbors()]
        def angles(p):
            return sorted(float(np.degrees(np.arccos(np.clip(np.dot(p[b]-p[a],p[c]-p[a])/np.linalg.norm(p[b]-p[a])/np.linalg.norm(p[c]-p[a]),-1,1)))) for b,c in itertools.combinations(nb,2))
        if len(nb)>1:angle_errors.extend(np.array(angles(x))-np.array(angles(reference)))
    return dict(bond_rms=float(np.sqrt(np.mean(np.array(bdiff)**2))) if bdiff else 0,
                angle_rms=float(np.sqrt(np.mean(np.array(angle_errors)**2))) if angle_errors else 0,
                angle_errors=angle_errors)


def measure():
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer
    from fit.parser import TMQMParser,_iter_xyz_blocks,_iter_bo_blocks,_assemble_raw,_attach_qm_properties
    from jacobik.topology.mapping import assemble
    from jacobik.builder.ligands import _build_ligand_groups,ligand_inputs_from_mol
    from tools.jdev.cases import Case
    from tools.jdev.start import _crystal_ligand_coords
    selection=json.loads((OUT/'selection.json').read_text());root=REPO/'data/refit/clean/tmQM';parser=TMQMParser(root)
    selected=selection['selected'];by_shard={}
    for r in selected:by_shard.setdefault(r['shard'],set()).add(r['code'])
    rows=[];failures=[]
    for shard,codes in by_shard.items():
        for (comment,atoms),(code,bonds) in zip(_iter_xyz_blocks(root/'xyz'/f'tmQM_{shard}.xyz'),_iter_bo_blocks(root/'bo'/f'tmQM_{shard}.BO')):
            if code not in codes:continue
            try:
                csv=parser.csv_index.get(code);mol=assemble(_assemble_raw(comment,atoms,code,bonds,csv))
                if csv is not None:_attach_qm_properties(mol,csv)
                groups=_build_ligand_groups(mol,mol.smiles);inputs=ligand_inputs_from_mol(mol,groups)
                case=Case(name=code,make_builder=None,charge=mol.charge,mult=mol.spin_state,tmqm_mol=mol,ligand_groups=groups,ligand_inputs=inputs,
                    ref_X=np.array([a.position for a in mol.atoms]))
                crystal=_crystal_ligand_coords(case)
                for item in [r for r in selected if r['shard']==shard and r['code']==code]:
                    smi=item['smiles'];dn=item['donors'];base=Chem.MolFromSmiles(smi)
                    rd,start,_=_smiles_to_conformer(smi,donors=dn,chelate_embed=True,embed='mmff')
                    reference=crystal[smi][0];assert len(reference)==len(start)
                    flat=sorted(set(i for g in dn for i in g));pairs=[(a,b) for j,a in enumerate(flat) for b in flat[j+1:]]
                    unbonded=[(a,b) for a,b in pairs if base.GetBondBetweenAtoms(a,b) is None]
                    sur,capx,mapping=cap_start(base,rd,start,dn)
                    variants={'bare_keep_pairs':refine(rd,start,pairs),'cap_keep_pairs':refine(sur,capx,pairs)[mapping]}
                    if unbonded!=pairs:
                        variants['bare_release_bonded']=refine(rd,start,unbonded)
                        variants['cap_release_bonded']=refine(sur,capx,unbonded)[mapping]
                    stats={k:errors(rd,x,reference,dn) for k,x in variants.items()}
                    row=dict(**item,metrics=stats);rows.append(row)
                    np.savez_compressed(OUT/f'{code}_{len(rows)}.npz',reference=reference,start=start,**variants)
                    print(code,item['family'],json.dumps(stats),flush=True)
            except Exception as exc:
                failures.append(dict(code=code,shard=shard,error=f'{type(exc).__name__}: {exc}'));print('FAILED',failures[-1],flush=True)
            (OUT/'results.json').write_text(json.dumps(dict(selection=selection,rows=rows,failures=failures,scope='Training-ligand geometry only; positive result would not establish complex-energy benefit. Both arms share original atom coordinates and donor constraints; cap retention isolated from releasing covalent donor constraints where relevant.'),indent=2)+'\n')
    print('Completed',len(rows),'failures',len(failures),flush=True)


if __name__=='__main__':
    setup();select() if 'select' in sys.argv else measure()
