"""Build D-B's crystal Q7 bound-angle sidecar from cal/tail training parents only."""
from __future__ import annotations
import json
from pathlib import Path
import sys

from tools.jgen import inputs
from fit.bends import extract_bound_angle_observations, write_bound_angle_sidecar

DATA_ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
WORKTREE=Path(__file__).resolve().parents[4]
PANEL=WORKTREE/'research/bench/benchmarks/panels/tmqm_cal_tail_120.json'
LOCKED=WORKTREE/'research/bench/benchmarks/locked_exclusions.json'
OUT=DATA_ROOT/'research/data/jgen/campaign/D/q7_bound_angles.bends.json'

def main():
    panel=json.loads(PANEL.read_text()); lock=json.loads(LOCKED.read_text())
    excluded=set(lock.get('reserve_all_codes',[]))|set(lock.get('locked_codes',[]))
    codes=[c for c in (sys.argv[1:] or panel['codes']) if c not in excluded]
    shards=inputs.frame_shards(); obs=[]; failures=[]
    for i,code in enumerate(codes,1):
        try:
            shard,dataset=shards[code]
            mol=inputs.load_mol(code,shard,dataset)
            obs.extend(extract_bound_angle_observations(mol,code))
        except Exception as exc:
            failures.append({'code':code,'error':f'{type(exc).__name__}: {exc}'[:400]})
        if i%10==0: print(f'{i}/{len(codes)} parents; {len(obs)} angles; {len(failures)} failures',flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    payload=write_bound_angle_sidecar(OUT,obs,{'source':'crystal cal/tail training frame','parents_requested':len(codes),
        'parents_ok':len(codes)-len(failures),'locked_exclusions_sha256':lock.get('locked_manifest_sha256')},min_n=3)
    (OUT.parent/'q7_bound_angles_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps({'n_parents':len(codes),'n_observations':len(obs),'n_entries':len(payload['entries']),
        'failures':failures[:20],'out':str(OUT)},indent=2))
if __name__=='__main__': main()
