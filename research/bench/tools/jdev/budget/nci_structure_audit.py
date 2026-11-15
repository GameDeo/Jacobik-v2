"""Trace ligand bond errors to input, embedding, or solver using exact topology."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, pool_path
from pools import setup, set_cases


def run():
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from jacobik.topology.graph import get_ligand_neighbors
    names = {c:(n,s) for s in ('dev32','n48') for c,n in set_cases(s)}
    selection = json.loads((REPO/'data/refit/reports/n2_targeted_rescore_2026-10-18.json').read_text())
    roots = [REPO/'data/refit/reports'/p for p in ('n2_energy_diagnosis_2026-10-18','n2_energy_diagnosis_extra_2026-10-18')]
    report = {}
    for code in ('JEFJUZ','XEPKAF','PEWBOL'):
        name,split = names[code]; case = get_case(name)
        with starting_geometry(case,'full_oracle'):
            mol,_,_ = case.make_builder()._build_shell(False,False)
        pos = {a.idx:i for i,a in enumerate(mol.atoms)}
        nb = get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
        edges = [(pos[a],pos[b]) for a in nb for b in nb[a] if a<b]
        poses = {'reference':np.array([a.position for a in mol.atoms])}
        for method in ('etkdg','mmff','gfn2'):
            m,_,_ = case.make_builder()._build_shell(False,False,chelate_embed=True,embed=method)
            assert [a.symbol for a in m.atoms]==[a.symbol for a in mol.atoms]
            poses['embed_'+method] = np.array([a.position for a in m.atoms])
        for row in selection['pools']:
            if row['case']!=code:continue
            with np.load(pool_path('etkdg_u1_os',split,code,row['seed'])) as z:
                for label,key in [('before','base_index'),('after','cand_index')]:
                    poses[f"s{row['seed']}_{label}"] = z['X'][row[key]]
        lengths = {label:np.array([np.linalg.norm(x[a]-x[b]) for a,b in edges]) for label,x in poses.items()}
        bonds=[]
        for j,(a,b) in enumerate(edges):
            bonds.append(dict(atoms=[a+1,b+1],symbols=[mol.atoms[a].symbol,mol.atoms[b].symbol],
                              lengths={k:float(v[j]) for k,v in lengths.items()},
                              max_pool_reference_error=float(max(abs(v[j]-lengths['reference'][j]) for k,v in lengths.items() if k.startswith('s')))))
        bonds.sort(key=lambda b:-b['max_pool_reference_error'])
        ligands=[]
        for lig,li in zip(mol.ligands,case.ligand_inputs):
            rows=[pos[a] for a in lig.atom_indices]
            ligands.append(dict(id=lig.ligand_id,smiles=li.smiles,donors=li.donors,atoms=[i+1 for i in rows]))
        report[code]=dict(ligands=ligands,bonds=bonds,
            maximum_bond_error={k:float(np.max(np.abs(v-lengths['reference']))) for k,v in lengths.items()},
            maximum_before_after_bond_change={str(s):float(np.max(np.abs(lengths[f's{s}_before']-lengths[f's{s}_after']))) for s in (11,12,13)})
        print(code,json.dumps({k:v for k,v in report[code].items() if k!='bonds'}),flush=True)
        print('worst bonds',json.dumps(bonds[:8]),flush=True)
    out=REPO/'data/refit/reports/n2_structure_audit_2026-10-18.json'
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    setup();run()
