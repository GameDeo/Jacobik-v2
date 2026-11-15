"""Separate, non-production embedding diagnostics with common starts.

No caps adopted, no complex optimization, no NCI score fitting. Cp and arene
restraint release is screened on independently selected training ligands.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import zlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO
from pools import setup,set_cases
from nci_cap_generality import refine,cap_start,errors
from nci_training_calibration import write,digest,stream

OUT=REPO/'data/refit/reports/n2_embedding_validation_2026-10-18'


def select():
    from rdkit import Chem
    from fit.pipeline.extract import benchmark_codes
    OUT.mkdir(exist_ok=True);excluded=benchmark_codes()
    source=REPO/'data/refit/reports/oxidation_training_modes_2026-10-18.json'
    data=json.loads(source.read_text());groups={k:{} for k in ('Cp','substituted_Cp','arene')}
    for row in data['rows']:
        assert row['code'] not in excluded
        for li in row['ligands']:
            smi=li['smiles'];dn=li['donors'];base=Chem.MolFromSmiles(smi)
            if base is None or '[H]' in smi or Chem.AddHs(base).GetNumAtoms()>80:continue
            face=[g for g in dn if len(g) in (5,6) and all(base.GetAtomWithIdx(i).GetAtomicNum()==6 and base.GetAtomWithIdx(i).GetIsAromatic() for i in g)]
            if len(face)!=1:continue
            g=face[0]
            if len(g)==5 and sum(base.GetAtomWithIdx(i).GetFormalCharge() for i in g)==-1:
                family='Cp' if base.GetNumHeavyAtoms()==5 else 'substituted_Cp'
            elif len(g)==6 and sum(base.GetAtomWithIdx(i).GetFormalCharge() for i in g)==0:family='arene'
            else:continue
            key=repr((smi,dn));groups[family].setdefault(key,dict(code=row['code'],shard=row['set'],dir=str(REPO/'data/refit/clean/tmQM'),smiles=smi,donors=dn,face=g,family=family,key=key))
    selected=[]
    for family,items in groups.items():
        selected.extend(sorted(items.values(),key=lambda r:zlib.crc32(('cp-release-2026-10-18:'+r['key']).encode()))[:8])
    write(OUT/'selection.json',dict(source_sha256=digest(source),counts={k:len(v) for k,v in groups.items()},selected=selected,
        selection='Up to eight unique SMILES/donor-pattern ligands per family, stable hash before measuring geometry; <=80 atoms, exactly one all-carbon aromatic eta5/eta6 face. All canonical benchmark exclusions. No eta2 or group-13 change.',
        arms='Same production MMFF starting geometry; strict GFN2 all donor pairs vs release only pairs within the selected eta5/eta6 face. Cross-site constraints retained; caps absent in both arms.'))
    # Broad metadata screen first, then graph match. Includes N-H anions and
    # aromatic negative N; no assumption about exact SMILES spelling.
    census=[];counts={};pattern=Chem.MolFromSmarts('[#7;-1]~[#7;-1]')
    for dataset in ('tmQM','lnQM'):
        path=REPO/f'data/refit/clean/{dataset}/tmQM_y.csv';n=0;screen=0;parse_fail=0;missing=0;matches=[]
        with path.open(encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh,delimiter=';'):
                code=row['CSD_code']
                if code in excluded:continue
                n+=1;smi=row.get('SMILES') or ''
                if not smi:missing+=1;continue
                if len(re.findall(r'\[[nN][^\]]*-(?!>)',smi))<2:continue
                screen+=1;rd=Chem.MolFromSmiles(smi,sanitize=False)
                if rd is None:parse_fail+=1;continue
                if rd.HasSubstructMatch(pattern):
                    matches.append(code);census.append(dict(dataset=dataset,code=code,smiles=smi))
        counts[dataset]=dict(nonbenchmark_csv_rows=n,missing_smiles=missing,prefiltered=screen,parse_failures=parse_fail,adjacent_negative_NN_codes=matches,sha256=digest(path))
    write(OUT/'anionic_NN_metadata_census.json',dict(datasets=counts,matches=census,
        scope='Complete cleaned CSV metadata, including missing-SMILES accounting. Graph screen identifies adjacent formally anionic N atoms; it does not yet prove both atoms are primary metal donors. No geometry or energy selection.'))
    print('Selected',json.dumps({k:sum(r['family']==k for r in selected) for k in groups}),'NN metadata',json.dumps(counts),flush=True)


def make_case(mol):
    from tools.jdev.cases import Case
    from jacobik.builder.ligands import _build_ligand_groups,ligand_inputs_from_mol
    groups=_build_ligand_groups(mol,mol.smiles);inputs=ligand_inputs_from_mol(mol,groups)
    return Case(name=mol.csd_code,make_builder=None,charge=mol.charge,mult=mol.spin_state,tmqm_mol=mol,ligand_groups=groups,ligand_inputs=inputs,ref_X=np.array([a.position for a in mol.atoms]))


def nn_census():
    import subprocess
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from jacobik.builder.ligands import _build_ligand_groups,ligand_inputs_from_mol
    census=json.loads((OUT/'anionic_NN_metadata_census.json').read_text());metadata=census['matches']
    codes={r['code'] for r in metadata};pattern=r'CSD_code = ('+'|'.join(sorted(codes))+r')\b'
    result=subprocess.run(['rg','--no-heading','-n',pattern,str(REPO/'data/refit/clean/tmQM/xyz')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    chosen=[]
    for line in result.stdout.splitlines():
        m=re.match(r'^(.*tmQM_([^\\/]+)\.xyz):\d+:.*CSD_code = (\w+)\b',line)
        assert m,line
        chosen.append(dict(code=m[3],shard=m[2],dir=str(REPO/'data/refit/clean/tmQM')))
    found={r['code'] for r in chosen};assert found<=codes
    rows=[];failures=[]
    for row,mol,error in stream(chosen):
        try:
            if error:raise ValueError(error)
            groups=_build_ligand_groups(mol,mol.smiles);inputs=ligand_inputs_from_mol(mol,groups)
            if inputs is None:raise ValueError('ligand inputs unavailable')
            for li in inputs:
                rd=Chem.MolFromSmiles(li.smiles)
                if rd is None:continue
                flat=set(i for g in li.donors for i in g)
                motif=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in rd.GetBonds()
                    if all(a.GetAtomicNum()==7 and a.GetFormalCharge()==-1 for a in (b.GetBeginAtom(),b.GetEndAtom()))]
                if not motif:continue
                core=MurckoScaffold.GetScaffoldForMol(rd)
                rows.append(dict(**row,smiles=li.smiles,donors=li.donors,motifs=motif,
                    both_primary=any(a in flat and b in flat for a,b in motif),
                    scaffold=Chem.MolToSmiles(core if core.GetNumAtoms() else rd,isomericSmiles=False)))
        except Exception as exc:failures.append(dict(code=row['code'],error=f'{type(exc).__name__}: {exc}'))
    primary=[r for r in rows if r['both_primary']]
    write(OUT/'anionic_NN_topology_census.json',dict(metadata_sha256=digest(OUT/'anionic_NN_metadata_census.json'),
        codes_missing_raw=sorted(codes-found),rows=rows,failures=failures,
        summary=dict(metadata_codes=len(codes),raw_codes=len(found),mapped_motif_ligands=len(rows),primary_motif_ligands=len(primary),
                     primary_unique_smiles=len({r['smiles'] for r in primary}),primary_unique_scaffolds=len({r['scaffold'] for r in primary})),
        scope='All metadata graph-screen matches checked against mapped primary donors, <=160 complex atoms. Mapping/size failures recorded. LnQM has no SMILES and cannot be excluded by this census. No embedding adoption.'))
    print('NN topology',json.dumps(dict(primary=len(primary),unique_smiles=len({r['smiles'] for r in primary}),unique_scaffolds=len({r['scaffold'] for r in primary}),failures=len(failures))),flush=True)


def training():
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer
    from tools.jdev.start import _crystal_ligand_coords
    selection=json.loads((OUT/'selection.json').read_text());chosen={}
    for r in selection['selected']:chosen[(r['shard'],r['code'])]=r
    rows=[];failures=[]
    for row,mol,error in stream(chosen.values()):
        try:
            if error:raise ValueError(error)
            case=make_case(mol);crystal=_crystal_ligand_coords(case)
            for r in [r for r in selection['selected'] if r['code']==row['code']]:
                smi=r['smiles'];dn=r['donors'];base=Chem.MolFromSmiles(smi)
                rd,start,_=_smiles_to_conformer(smi,donors=dn,chelate_embed=True,embed='mmff')
                reference=crystal[smi][0];assert len(reference)==len(start)
                flat=sorted(set(i for g in dn for i in g));face=set(r['face'])
                pairs=[(a,b) for j,a in enumerate(flat) for b in flat[j+1:]]
                released=[(a,b) for a,b in pairs if not (a in face and b in face)]
                variants=dict(keep_pairs=refine(rd,start,pairs),release_face=refine(rd,start,released))
                result=dict(**r,metrics={k:errors(rd,x,reference,dn) for k,x in variants.items()})
                rows.append(result);np.savez_compressed(OUT/f"{r['family']}_{r['code']}_{len(rows)}.npz",reference=reference,start=start,**variants)
                print(r['code'],r['family'],json.dumps(result['metrics']),flush=True)
        except Exception as exc:
            failures.append(dict(code=row['code'],error=f'{type(exc).__name__}: {exc}'));print('FAILED',failures[-1],flush=True)
        write(OUT/'training_results.json',dict(selection=selection,rows=rows,failures=failures))


def jef():
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer
    from tools.jdev.cases import get_case
    from tools.jdev.start import _crystal_ligand_coords
    name=next(n for s in ('dev32','n48') for c,n in set_cases(s) if c=='JEFJUZ');case=get_case(name)
    crystal=_crystal_ligand_coords(case);rows=[]
    for li in case.ligand_inputs:
        smi=li.smiles;dn=li.donors;base=Chem.MolFromSmiles(smi)
        if '[N-]' not in smi and not any(len(g)>=5 for g in dn):continue
        if any(r['smiles']==smi for r in rows):continue
        rd,start,_=_smiles_to_conformer(smi,donors=dn,chelate_embed=True,embed='mmff')
        reference=crystal[smi][0];flat=sorted(set(i for g in dn for i in g))
        pairs=[(a,b) for j,a in enumerate(flat) for b in flat[j+1:]]
        released=[(a,b) for a,b in pairs if base.GetBondBetweenAtoms(a,b) is None]
        variants=dict(bare_keep_pairs=refine(rd,start,pairs))
        if '[N-]' in smi:
            sur,x,mapping=cap_start(base,rd,start,dn)
            variants.update(cap_keep_pairs=refine(sur,x,pairs)[mapping],bare_release_bonded=refine(rd,start,released),cap_release_bonded=refine(sur,x,released)[mapping])
        else:variants['bare_release_face']=refine(rd,start,[])
        result=dict(smiles=smi,donors=dn,metrics={k:errors(rd,x,reference,dn) for k,x in variants.items()});rows.append(result)
        np.savez_compressed(OUT/f'JEFJUZ_factorial_{len(rows)}.npz',reference=reference,start=start,**variants)
        print('JEFJUZ',smi,json.dumps(result['metrics']),flush=True)
        write(OUT/'JEFJUZ_factorial.json',dict(rows=rows,scope='Common production MMFF coordinates, identical original atoms in capped arm. Geometry-only factorial, no runtime change or new complex solve.'))


if __name__=='__main__':
    setup();p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','training','jef','nn-census']);args=p.parse_args()
    {'select':select,'training':training,'jef':jef,'nn-census':nn_census}[args.stage]()
