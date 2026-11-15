"""Repair saved FK exports and recompute unrelaxed energies, without any solve/embed/relaxation."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rdkit import Chem
from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs
from tools.jgen.atom_export import symbols_for_conformer
from tools.runkit import mace,resources


def builder_shell(code):
    """Reconstruct the engine's atom graph only. No conformer generation or optimisation."""
    from jacobik.builder import engine
    sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
    builder,_,err=build_builder_and_inputs(mol)
    if builder is None:raise RuntimeError(err)
    def graph_only(smiles,**kw):
        rd=Chem.AddHs(Chem.MolFromSmiles(smiles))
        return rd,np.zeros((rd.GetNumAtoms(),3)),('[H]' in smiles or '[H' in smiles)
    with patch.object(engine,'_smiles_to_conformer',graph_only):
        shell,_,_=engine._build_complex_shell(builder.metal,builder.charge,builder.spin,builder.geometry,builder.ligands)
    from types import SimpleNamespace
    conf=SimpleNamespace(complex=shell)
    symbols=symbols_for_conformer(conf)
    assert symbols == [a.symbol for a in conf.complex.atoms]
    return mol,conf,symbols


def backup(path):
    old=path.with_name(path.stem+'.before_atom_order_fix.json')
    if not old.exists():old.write_bytes(path.read_bytes())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    report={'scope':'unrelaxed single points only; no solves, embeds or relaxations','files':{},'failures':{},'symbols':{}}
    shells={}
    def shell(code):
        if code not in shells:
            shells[code]=builder_shell(code)
            mol,conf,sy=shells[code]
            report['symbols'][code]={'builder_symbols':sy,'frame_mismatches':sum(a.symbol!=s for a,s in zip(mol.atoms,sy)),
                'provenance':'engine._build_complex_shell with graph-only RDKit helper, no embedding; atom list retained by _assemble_conformers'}
        return shells[code]
    with resources.gpu_lock('CodexB'):
        ev=mace.Evaluator('cuda',dtype='float64',threads=1)
        for name in ('o1_g2_cap30_unrel_gpu_results.json','g3_polish_sanity_complete.json','g3prime_sanity.json'):
            path=root/name;backup(path);d=json.loads(path.read_text());before=json.loads(path.with_name(path.stem+'.before_atom_order_fix.json').read_text());stats=[]
            for code,parent in d['parents'].items():
                try:
                    mol,conf,sy=shell(code);structs=[];owners=[]
                    for row in parent['rows']:
                        assert len(sy)==len(row.get('X_init_A',row.get('X0_A')))
                        if 'X_star_A' in row:
                            row['atom_symbols']=sy;row['symbols_order']='builder/conf.complex.atoms'
                            assert row['atom_symbols']==[a.symbol for a in conf.complex.atoms]
                            positions=[row['X_init_A'],row['X_star_A']]
                        else:
                            row['symbols']=sy;row['symbols_order']='builder/conf.complex.atoms'
                            assert row['symbols']==[a.symbol for a in conf.complex.atoms]
                            positions=[row['X0_A']]+[a['X_A'] for a in row['arms'].values()]
                        owners.append((row,len(structs),len(positions)))
                        structs.extend((sy,X,int(mol.charge),int(mol.spin_state)) for X in positions)
                    energies=ev.energies_kj(structs)
                    for row,start,n in owners:
                        if 'X_star_A' in row:
                            row['E_unrel_init_kj']=float(energies[start]);row['E_unrel_star_kj']=float(energies[start+1]);row['delta_E_unrel_kj']=float(energies[start+1]-energies[start])
                            row['xtb_validity']='VOID: historical relaxation atom labels were in frame order'
                            row['strain_status']='VOID: atom-order mismatch; no replacement relaxation performed'
                            row['classification']='score_only; historical strain classification void'
                        else:
                            for j,a in enumerate(row['arms'].values(),1):a['delta_delta_E_unrel_kj']=float(energies[start+j]-energies[start])
                    stats.append(code)
                except Exception as e:report['failures'][f'{name}/{code}']=repr(e)
                d['atom_order_correction']={'date':'2026-10-29','symbols_order':'builder/conf.complex.atoms','old_values':path.stem+'.before_atom_order_fix.json','xtb':'VOID; not rerun'}
                path.write_text(json.dumps(d,indent=2))
            def med(data):
                rows=[r for p in data['parents'].values() for r in p['rows']]
                if rows and 'X_star_A' in rows[0]:return {'LBFGS':float(np.median([r['delta_E_unrel_kj'] for r in rows]))}
                return {a:float(np.median([r['arms'][a]['delta_delta_E_unrel_kj'] for r in rows])) if all('delta_delta_E_unrel_kj' in r['arms'][a] for r in rows) else None for a in rows[0]['arms']}
            report['files'][name]={'parents_recomputed':stats,'old_median_energy_changes_kj':med(before),'corrected_median_energy_changes_kj':med(d)}
            print(name,report['files'][name],flush=True)
        # These exports explicitly map FK coordinates into reference order before
        # pairing them with frame labels. Recompute every SAVED top1 to verify.
        for folder in ('g3_trope/C1','g3_trope/P1','g3_trope/P2','o2_trope_r5/control_unrel','o2_trope_r5/adam_metric_unrel','o2_trope_r5/fire_metric_unrel'):
            for path in sorted((root/folder).glob('*.json')):
                d=json.loads(path.read_text())
                if 'per_seed' not in d:continue
                backup(path);code=d['code'];sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
                sy=[a.symbol for a in mol.atoms];Xref=[a.position for a in mol.atoms]
                rows=[(s,r) for s,r in d['per_seed'].items() if 'top1_X' in r]
                if not rows:continue
                try:
                    assert all(r['atom_symbols']==sy for _,r in rows)
                    es=ev.energies_kj([(sy,Xref,mol.charge,mol.spin_state)]+[(sy,r['top1_X'],mol.charge,mol.spin_state) for _,r in rows])
                    values={s:{'old':r['dE_unrel'],'corrected':float(es[i+1]-es[0])} for i,(s,r) in enumerate(rows)}
                    report['files'][str(path.relative_to(root))]={'top1_recomputed':values,'coordinate_order':'reference, explicitly mapped by suite.py; frame labels appropriate','full_row_coordinates':'not exported; previous floor/obscene metrics use same valid mapping'}
                    for i,(s,r) in enumerate(rows):r['dE_unrel']=float(es[i+1]-es[0])
                    d['top1_energy_recheck']=report['files'][str(path.relative_to(root))];path.write_text(json.dumps(d,indent=2))
                except Exception as e:report['failures'][str(path.relative_to(root))]=repr(e)
    # Preserve all historical values but make invalidity explicit on their roots.
    for name in ('o1_g2_results.json','o1_g2_cap30_results.json','o1_g2_cap30_selected_relax.json'):
        path=root/name
        if path.exists():
            backup(path);d=json.loads(path.read_text());d['energy_validity']='VOID: atom-order audit; xTB results not reusable, no replacement relaxation';path.write_text(json.dumps(d,indent=2))
    report['missing_coordinates']='O1 primary/cancelled wider-cap JSON lacks coordinates; xTB work inputs were not retained. Those historical energies stay void; the saved canonical O1 cap30 48-row comparison is recomputed.'
    (root/'atom_order_repair_report.json').write_text(json.dumps(report,indent=2));print('failures',report['failures'],flush=True)

if __name__=='__main__':main()
