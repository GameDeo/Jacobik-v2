"""Batched unrelaxed MACE and explicit G-3 score-gate accounting."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.runkit import mace, resources


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--energy',action='store_true')
    args=ap.parse_args();root=Path(args.root)
    # Reproduce the initial two-parent calibration without another solve.
    h=json.loads((root/'g3_fire_calibration_workers3.json').read_text())
    u=json.loads((root/'g3_fire_calibration_exact_rows.json').read_text())
    probe=[r['first_step_rms_A'] for r in h['parents']['HEDJOP']['rows']]+[r['first_step_rms_A'] for r in u['parents']['USAPOU']['rows']]
    median=float(np.median(probe));dt=float(.1*np.sqrt(.02/median))
    calibration={'parents':['HEDJOP','USAPOU'],'rows':12,'probe_dt':.1,
        'median_first_step_rms_A':median,'dt_start':dt,'dt_max':10*dt,'rms_rows_A':probe}
    (root/'g3_fire_calibration_combined.json').write_text(json.dumps(calibration,indent=2))
    original=root/'g3_polish_sanity.json'
    d=json.loads(original.read_text())
    retry=root/'g3_sanity_layout8.json'
    if retry.exists():
        extra=json.loads(retry.read_text())
        for code,parent in extra['parents'].items():
            d['parents'][code]=parent
            d['failures'].pop(code,None)
    d['source_files']=[original.name,retry.name]
    p=root/'g3_polish_sanity_complete.json'
    if p.exists():
        d=json.loads(p.read_text())
    p.write_text(json.dumps(d,indent=2))
    rows=[r for parent in d['parents'].values() for r in parent['rows']]
    if args.energy:
        with resources.gpu_lock('CodexB'):
            ev=mace.Evaluator('cuda',dtype='float64',threads=1)
            for code,parent in d['parents'].items():
                structs=[]
                for r in parent['rows']:
                    for X in [r['X0_A']]+[r['arms'][a]['X_A'] for a in ('adam_small','fire')]:
                        structs.append((r['symbols'],X,r['charge'],r['spin']))
                try:
                    E=ev.energies_kj(structs)
                    for i,r in enumerate(parent['rows']):
                        for j,a in enumerate(('adam_small','fire'),1):
                            r['arms'][a]['delta_delta_E_unrel_kj']=float(E[3*i+j]-E[3*i])
                except Exception as e:
                    parent['energy_failure']=f'{type(e).__name__}: {e}'
                p.write_text(json.dumps(d,indent=2))
    result={'complete':len(rows)==48 and not d['failures'],'polish':{},'failures':d['failures']}
    for a in ('adam_small','fire'):
        arm=[r['arms'][a] for r in rows]
        result['polish'][a]={'n':len(arm),'median_fraction':float(np.median([r['fraction'] for r in arm])),
            'median_gain':float(np.median([r['gain'] for r in arm])),
            'pass':len(arm)==48 and not d['failures'] and np.median([r['gain'] for r in arm])>=0 and np.median([r['fraction'] for r in arm])>=.25}
        energy=[r['delta_delta_E_unrel_kj'] for r in arm if 'delta_delta_E_unrel_kj' in r]
        if energy:
            result['polish'][a].update(energy_rows=len(energy),median_delta_delta_E_unrel_kj=float(np.median(energy)))
    c=json.loads((root/'g3_c0_sanity.json').read_text());a=json.loads((root/'g3_a1_sanity.json').read_text())
    common=sorted(set(c['parents'])&set(a['parents']))
    cs=[c['parents'][k]['top1_score'] for k in common];ass=[a['parents'][k]['top1_score'] for k in common]
    result['A1']={'parents':len(common),'median_C0':float(np.median(cs)),'median_A1':float(np.median(ass)),
        'median_paired_change':float(np.median(np.array(ass)-cs)),
        'pass':len(common)==8 and not c['failures'] and not a['failures'] and np.median(ass)>=np.median(cs)}
    (root/'g3_sanity_summary.json').write_text(json.dumps(result,indent=2,default=bool));print(json.dumps(result,indent=2,default=bool))

if __name__=='__main__':
    main()
