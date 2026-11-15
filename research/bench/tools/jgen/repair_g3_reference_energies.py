"""Add reference-relative energies to corrected saved O1/polish rows; no solves."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.jgen.repair_g3_atom_order import builder_shell,backup
from tools.runkit import mace,resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
    origin=json.loads((root/'o1_g2_cap30_unrel_gpu_results.json').read_text())
    prior=json.loads((root/'o1_g2_cap30_unrel_gpu_results.before_atom_order_fix.json').read_text())
    report=json.loads((root/'atom_order_repair_report.json').read_text());comparison={}
    with resources.gpu_lock('CodexB'):
        ev=mace.Evaluator('cuda',dtype='float64',threads=1)
        for code,parent in origin['parents'].items():
            mol,conf,sy=builder_shell(code)
            ref=float(ev.energies_kj([([a.symbol for a in mol.atoms],[a.position for a in mol.atoms],mol.charge,mol.spin_state)])[0])
            parent['reference_energy_kj']=ref
            atom_pos={a.idx:i for i,a in enumerate(conf.complex.atoms)}
            donors={atom_pos[d] for lig in conf.complex.ligands for d in lig.donor_atoms}
            metal=atom_pos[conf.complex.metal_idx];non=[i for i in range(len(sy)) if i!=metal and i not in donors]
            oldrows={tuple(r['key']):r for r in prior['parents'][code]['rows']}
            comparison[code]={}
            for row in parent['rows']:
                row['C0_delta_E_unrel_vs_reference_kj']=row['E_unrel_init_kj']-ref
                row['LBFGS_delta_E_unrel_vs_reference_kj']=row['E_unrel_star_kj']-ref
                X=np.asarray(row['X_star_A']);contact=float(np.linalg.norm(X[non]-X[metal],axis=-1).min()) if non else None
                row['closest_non_donor_metal_contact_A']=contact;row['metal_contact_lt_1_2A']=contact is not None and contact<1.2
            top=min(parent['rows'],key=lambda r:r['rank']);old=oldrows[tuple(top['key'])]
            comparison[code]['C0']={'old_top1':old['E_unrel_init_kj']-ref,'corrected_top1':top['C0_delta_E_unrel_vs_reference_kj']}
    (root/'o1_g2_cap30_unrel_gpu_results.json').write_text(json.dumps(origin,indent=2))
    for name in ('g3_polish_sanity_complete.json','g3prime_sanity.json'):
        p=root/name;d=json.loads(p.read_text());before=json.loads(p.with_name(p.stem+'.before_atom_order_fix.json').read_text())
        for code,parent in d['parents'].items():
            base={tuple(r['key']):r for r in origin['parents'][code]['rows']};oldbase={tuple(r['key']):r for r in prior['parents'][code]['rows']}
            oldrows={tuple(r['key']):r for r in before['parents'][code]['rows']};ref=origin['parents'][code]['reference_energy_kj']
            for row in parent['rows']:
                initial=base[tuple(row['key'])]['E_unrel_init_kj'];row['C0_delta_E_unrel_vs_reference_kj']=initial-ref
                for a,arm in row['arms'].items():arm['delta_E_unrel_vs_reference_kj']=initial+arm['delta_delta_E_unrel_kj']-ref
            top=next(r for r in parent['rows'] if tuple(r['key'])==tuple(min(origin['parents'][code]['rows'],key=lambda r:r['rank'])['key']))
            for a,arm in top['arms'].items():
                oldarm=oldrows[tuple(top['key'])]['arms'][a]
                olddelta=oldarm.get('delta_delta_E_unrel_kj')
                comparison[code][name+'/'+a]={'old_top1':None if olddelta is None else oldbase[tuple(top['key'])]['E_unrel_init_kj']+olddelta-ref,
                                             'corrected_top1':arm['delta_E_unrel_vs_reference_kj']}
        p.write_text(json.dumps(d,indent=2))
    report['reference_relative_top1_by_parent_kj']=comparison
    report['A1']='G-3a A1 was score-only and stopped on its score gate; no A1 coordinates or energy evaluation was produced.'
    report['C0']='Original 48 O1 baseline coordinates recomputed above; reference-relative top1 energies included.'
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    (root/'atom_order_repair_report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    path=root/'o2_trope_r5/adam_metric_top1_relax.json'
    if path.exists():
        backup(path);d=json.loads(path.read_text());d['energy_validity']='VOID per owner atom-order review; historical xTB not reused or rerun';path.write_text(json.dumps(d,indent=2))
    print('reference energies and contacts corrected for',len(comparison),'parents',flush=True)

if __name__=='__main__':main()
