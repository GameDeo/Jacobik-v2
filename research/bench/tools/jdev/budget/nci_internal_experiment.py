"""Ligand-only repair hypotheses, with fixed-pose complex energy measurements.

Experimental tool only: no production monkeypatch, no reference-guided solver,
no complex relaxation. References enter only diagnostic oracle comparisons.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup,set_cases
from energy_viewer import align_transform

OUT=REPO/'data/refit/reports/n2_internal_experiment_2026-10-18'


def bridge_lengths(rdmol,x,target):
    """Change only heavy-atom bridge lengths; preserve every bond angle and torsion.

    x and target have identical atom order. Ring bonds cannot be stretched this
    way. Moving a whole component parallel to its bridge leaves its geometry
    and both endpoint bond directions unchanged.
    """
    import networkx as nx
    g=nx.Graph();g.add_nodes_from(range(rdmol.GetNumAtoms()))
    g.add_edges_from((b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in rdmol.GetBonds())
    y=np.array(x,copy=True)
    for a,b in list(nx.bridges(g)):
        if rdmol.GetAtomWithIdx(a).GetAtomicNum()==1 or rdmol.GetAtomWithIdx(b).GetAtomicNum()==1:continue
        sub=g.copy();sub.remove_edge(a,b)
        side=nx.node_connected_component(sub,b)
        unit=y[b]-y[a];length=np.linalg.norm(unit);unit/=length
        shift=np.linalg.norm(target[b]-target[a])-length
        y[sorted(side)]+=shift*unit
    return y


def trial_ligand(smiles,donors,smiles_to_conformer=None):
    """Retain valence caps through xTB; free intradonor covalent distances.

    Scope here is Cp and H-cappable lone-pair anions. No target coordinates.
    The generic production refinement otherwise deliberately holds donor pairs.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from jacobik.builder.embed import cap_anions,map_surrogate_positions
    from jacobik.builder.engine import _gfn2_refine,_smiles_to_conformer
    base=Chem.MolFromSmiles(smiles);orig=Chem.AddHs(base)
    cap,caps=cap_anions(base,donors)
    if caps:
        assert all(cap.GetAtomWithIdx(i).GetAtomicNum()==1 for i in caps),'H caps only in this experiment'
        mh=Chem.AddHs(cap);p=AllChem.ETKDGv3();p.randomSeed=42
        assert AllChem.EmbedMolecule(mh,p)>=0
        AllChem.MMFFOptimizeMolecule(mh,maxIters=2000)
        x=mh.GetConformer().GetPositions()
        # Adjacent donor distances are covalent internals, not a chelate bite constraint.
        flat=[i for g in donors for i in g]
        assert all(base.GetBondBetweenAtoms(a,b) is not None for j,a in enumerate(flat) for b in flat[j+1:])
        y=_gfn2_refine(mh,x,None)
        assert np.linalg.norm(y-x)>1e-4,'GFN2 did not refine the capped ligand (possible fallback)'
        return map_surrogate_positions(orig,mh,y),dict(caps=caps,method='H-capped ligand GFN2, no adjacent-donor restraint')
    assert len(donors)==1 and len(donors[0])>=3
    mh,x,_=(smiles_to_conformer or _smiles_to_conformer)(smiles,donors=donors,chelate_embed=True,embed='mmff')
    y=_gfn2_refine(mh,x,None)
    assert np.linalg.norm(y-x)>1e-4,'GFN2 did not refine the haptic ligand (possible fallback)'
    return y,dict(caps=[],method='uncapped haptic ligand GFN2, no intra-face restraint')


def embeddings():
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from jacobik.topology.graph import get_ligand_neighbors
    names={c:n for s in ('dev32','n48') for c,n in set_cases(s)}
    OUT.mkdir(exist_ok=True);report={}
    for code in ('JEFJUZ','XEPKAF'):
        case=get_case(names[code]);mol,_,_=case.make_builder()._build_shell(False,False,chelate_embed=True)
        with starting_geometry(case,'full_oracle'):
            ref,_,_=case.make_builder()._build_shell(False,False)
        pos={a.idx:i for i,a in enumerate(mol.atoms)}
        nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
        arrays={};ligands=[]
        for lig,li in zip(mol.ligands,case.ligand_inputs):
            if len(li.donors[0])<3 and '[N-]' not in li.smiles:continue
            x,info=trial_ligand(li.smiles,li.donors)
            rows=[pos[a] for a in lig.atom_indices];assert len(x)==len(rows)
            local={a:j for j,a in enumerate(lig.atom_indices)}
            edges=[(local[a],local[b]) for a in lig.atom_indices for b in nb[a] if b in local and a<b]
            reference=np.array([ref.atoms[i].position for i in rows]);original=np.array([mol.atoms[i].position for i in rows])
            length=lambda p:np.array([np.linalg.norm(p[a]-p[b]) for a,b in edges])
            key=f'L{lig.ligand_id}';arrays[key]=x
            ligands.append(dict(key=key,rows=rows,symbols=[mol.atoms[i].symbol for i in rows],smiles=li.smiles,donors=li.donors,info=info,
                base_bond_rms=float(np.sqrt(np.mean((length(original)-length(reference))**2))),
                trial_bond_rms=float(np.sqrt(np.mean((length(x)-length(reference))**2))),
                bonds=[dict(atoms=[rows[a]+1,rows[b]+1],reference=float(r),base=float(o),trial=float(v)) for (a,b),r,o,v in zip(edges,length(reference),length(original),length(x))]))
        np.savez_compressed(OUT/f'{code}_embeds.npz',**arrays)
        report[code]=dict(ligands=ligands)
        print(code,json.dumps(report[code]),flush=True)
    (OUT/'embeddings.json').write_text(json.dumps(report,indent=2)+'\n')


def evaluate():
    from rdkit import Chem
    from jacobik.core.energy import MACEPolarEvaluator
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    names={c:n for s in ('dev32','n48') for c,n in set_cases(s)}
    model=REPO/'tools/jdev/MACE-POLAR-1-S.model'
    ev=MACEPolarEvaluator(model='polar-1-s',device='cuda',dtype='float64',model_path=model,threads=1)
    info=json.loads((OUT/'embeddings.json').read_text());roots=[REPO/'data/refit/reports'/r for r in ('n2_energy_diagnosis_2026-10-18','n2_energy_diagnosis_extra_2026-10-18')]
    report=[]
    for code,item in info.items():
        case=get_case(names[code])
        with starting_geometry(case,'full_oracle'):
            mol,_,_=case.make_builder()._build_shell(False,False)
        symbols=[a.symbol for a in mol.atoms]
        with np.load(roots[0]/code/'reference.npz') as z:reference=z['X']
        with np.load(OUT/f'{code}_embeds.npz') as z:embeds={k:z[k] for k in z.files}
        ligands=item['ligands'];cp=[l for l in ligands if len(l['donors'][0])>=3];other=[l for l in ligands if l not in cp]
        groups={'cp':cp,'anionic_kappa':other,'both':ligands}
        for label in ('reference',*[f's{s}_{k}' for s in (11,12,13) for k in ('before','after')]):
            stem=next(r/code/label for r in roots if (r/code/label).with_suffix('.npz').exists())
            with np.load(stem.with_suffix('.npz')) as z:x=z['X']
            poses={'original':x}
            for group,ls in groups.items():
                if not ls:continue
                y=x.copy();oracle=x.copy()
                for l in ls:
                    rows=l['rows'];heavy=[j for j,s in enumerate(l['symbols']) if s!='H']
                    coords=embeds[l['key']];R,t=align_transform(coords,x[rows],heavy);y[rows]=coords@R+t
                    rc=reference[rows];R,t=align_transform(rc,x[rows],heavy);oracle[rows]=rc@R+t
                poses['trial_'+group]=y;poses['oracle_'+group]=oracle
            if other:
                for tag in ('trial','oracle'):
                    y=x.copy()
                    for l in other:
                        rd=Chem.AddHs(Chem.MolFromSmiles(l['smiles']));target=embeds[l['key']] if tag=='trial' else reference[l['rows']]
                        y[l['rows']]=bridge_lengths(rd,x[l['rows']],target)
                    poses['bond_'+tag]=y
                    poses['bond_'+tag+'_plus_cp']=poses['trial_cp']+(y-x)
            energy=ev.energies([(symbols,p,case.charge,case.mult) for p in poses.values()])*96.4853321233
            row=dict(case=code,pose=label,changes={k:float(e-energy[0]) for k,e in zip(poses,energy)})
            report.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(OUT/f'{code}_{label}_counterfactuals.npz',**poses)
        (OUT/'energies.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['embed','energy']);args=p.parse_args()
    setup();embeddings() if args.stage=='embed' else evaluate()
