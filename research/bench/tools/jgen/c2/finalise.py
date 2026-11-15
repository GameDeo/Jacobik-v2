"""Rebuild timing/identity from the raw ledger after both writers finish.

This never re-solves or changes energies/coordinates. In particular, a wide
single-seed experiment pays the full preparation cost even on the native
E/Z path, whose preparation occurs inside the arm rather than shared setup.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def finalise(root):
    settings=json.loads(Path('research/campaigns/c2/C2_SETTINGS.json').read_text())
    arms=json.loads((root/'arms.json').read_text())
    ledger=json.loads((root/'timing.json').read_text())
    nseeds=len(settings['seeds'])
    checks=[]
    for code, phases in ledger['parent_phases'].items():
        shared=sum(phases.get('prefetch',[]))+sum(phases.get('prepare',[]))
        for arm in arms:
            name=arm['arm'];wide=arm['seed_budget']=='single_w48'
            seed=sum(phases.get(name+'_seed_rows',[]))
            reused=not wide and not any(arm['flags'].get(k,False) for k in ['start_kabsch'])
            if reused:seed+=sum(phases.get('seed_rows',[]))
            prepare=sum(phases.get(name+'_prepare',[]))
            btime=ledger.get('beam_times',{}).get(code,{})
            rawbeam=btime.get(name,btime.get('shared',[]))
            beam=sum(rawbeam) if isinstance(rawbeam,list) else (rawbeam or 0.)
            for path in (root/name/'records').glob(f'{code}_*.json'):
                record=json.loads(path.read_text())
                solve=float(record.get('wall_solve_s',0.))
                wall=shared+prepare+(seed+solve)/nseeds if wide else (shared+prepare+seed+solve)/nseeds
                record['c2_timing']=dict(shared_prepare_s=shared,per_arm_prepare_s=prepare,
                    seed_rows_s=seed,solve_assembly_s=solve,beam_wall_s=beam,end_to_end_s=wall,
                    partial=record.get('status')!='ok',
                    allocation='Full prepare per independent wide seed, including native E/Z; ordinary three-seed preparation charged once')
                record['c2_identity']=dict(seed_budget=arm['seed_budget'],W=48 if wide else 16,
                    initial_digest=ledger['initial_digests'].get(code,{}).get(name,ledger['initial_digests'].get(code,{}).get('shared')),
                    row_steps=ledger['row_steps'].get(code,{}).get(name,[]))
                temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(record));temp.replace(path)
            if wide:
                rows=ledger['row_steps'].get(code,{})
                control=sum(x['row_steps'] for x in rows.get('C',[]))
                wide_steps=sum(x['row_steps'] for x in rows.get(name,[]))/nseeds
                ratio=wide_steps/control if control and wide_steps else None
                checks.append(dict(code=code,arm=name,control_three_seed_steps=control,
                    wide_single_seed_steps=wide_steps,ratio=ratio,
                    passed=abs(ratio-1)<=settings['row_steps_tolerance'] if ratio is not None else None))
    (root/'compute_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Compute mismatches:',[x for x in checks if x['passed'] is False])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();finalise(a.root)


if __name__=='__main__':main()
