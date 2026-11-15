"""Veto completed unguarded cal inputs with logged/cached GFN2 fallbacks.

Preserves raw records outside records/ before replacing canonical records with
explicit failures. No optimization, embedding retry, energy repair, or tuning.
"""
import argparse
import json
from pathlib import Path
from rdkit import Chem
from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs


def normalize(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol is not None else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--log',type=Path,required=True);ap.add_argument('--contract',type=Path,required=True)
    args=ap.parse_args();text=args.log.read_text();failures=set()
    for line in text.splitlines():
        if 'GFN2 embed refinement failed for ' in line:
            failures.add(line.split('GFN2 embed refinement failed for ',1)[1].split(' (',1)[0])
    normalized={normalize(s):s for s in failures};assert None not in normalized
    contract=json.loads(args.contract.read_text());matches={s:[] for s in failures};build_failures={}
    if failures:
        for code in contract['cal']['parents']:
            try:
                sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds);b,_,err=build_builder_and_inputs(mol)
                if b is None:build_failures[code]=err;continue
                for lig in b.ligands:
                    key=normalize(lig.smiles)
                    if key in normalized:matches[normalized[key]].append(code)
            except Exception as exc:build_failures[code]=str(exc)
        assert all(matches.values()),('unmapped failed embed',matches)
    invalid={code for codes in matches.values() for code in codes};changed=[];backup=args.root/'invalid_gfn2_originals'
    for path in args.root.rglob('records/*.json'):
        row=json.loads(path.read_text());code=row['code']
        if code not in invalid or row.get('input_validity')=='invalid_gfn2_embed_fallback':continue
        backup.mkdir(exist_ok=True);saved=backup/f'{row["arm"]}_{path.name}'
        if not saved.exists():saved.write_text(path.read_text())
        row.update(status='failed',reason='GFN2 embed fallback observed for a source ligand; generated structures and energies invalid',
                   top1_dE=None,floor_dE=None,n_rows=0,n_finite=0,input_validity='invalid_gfn2_embed_fallback',
                   invalidation_evidence={'log':str(args.log),'failed_ligands':[s for s,codes in matches.items() if code in codes],'raw_record_backup':str(saved)})
        for key in ('energies_kj','scores','row_keys','ref_energy_kj','top1_score','top1_bend_prior','floor_bend_prior'):row.pop(key,None)
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(row));temp.replace(path);changed.append(str(path))
    audit={'source_log':str(args.log),'failed_smiles':sorted(failures),'cohort_ligand_matches':{s:sorted(set(c)) for s,c in matches.items()},
           'input_build_failures':build_failures,'invalid_parents':sorted(invalid),'replaced_records':changed,'no_resolve_or_retry':True}
    (args.root/'gfn2_validity_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
