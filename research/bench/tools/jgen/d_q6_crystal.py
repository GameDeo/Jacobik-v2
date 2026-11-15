"""Crystal-frame Q6 change under D-B physical-all flag (CPU, panel cal/tail minus locked)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch

from tools.jgen import inputs
from tools.jdev.budget.common import stats_from_db
from jacobik.ik.geometry import GeometryExtractor
from jacobik.ik import geometry
from jacobik.stats import steric as q6
from jacobik.stats import keys as kb
from jacobik.topology import graph

DATA_ROOT = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
WORKTREE = Path(__file__).resolve().parents[4]
PANEL = WORKTREE / 'research/bench/benchmarks/panels/tmqm_cal_tail_120.json'
LOCKED = WORKTREE / 'research/bench/benchmarks/locked_exclusions.json'
OUT = DATA_ROOT / 'research/data/jgen/campaign/D/q6_crystal_weight14_zero.jsonl'
torch.set_num_threads(1)

def main():
    p=json.loads(PANEL.read_text()); locked=json.loads(LOCKED.read_text())
    excluded=set(locked.get('reserve_all_codes',[])) | set(locked.get('locked_codes',[]))
    codes=[c for c in (sys.argv[1:] or p['codes']) if c not in excluded]
    shards=inputs.frame_shards(); stats=stats_from_db(inputs.DB_PATH)
    records=[]; failures=[]
    for ix,code in enumerate(codes,1):
        try:
            shard,dataset=shards[code]; mol=inputs.load_mol(code,shard,dataset)
            X=torch.tensor(np.array([a.position for a in mol.atoms]),dtype=torch.float32)[None]
            sep=q6.ligand_separation(mol)
            rows={}
            for label,on in [('control',False),('q6_physical_all',True)]:
                graph.Q6_PHYSICAL_ALL=on; q6.Q6_PHYSICAL_ALL=on; geometry.Q6_PHYSICAL_ALL=on
                q6.Q6_WEIGHT_14=0.0
                ge=GeometryExtractor(mol,torch.device('cpu'))
                mu=[]; sigma=[]
                for key in ge._q6_elem_keys:
                    d=stats.dist('Q6',key)
                    mu.append(float(d.mu if d else 3.4)); sigma.append(float(d.sigma if d else .3))
                mu=torch.tensor(mu,dtype=torch.float32); sigma=torch.tensor(sigma,dtype=torch.float32)
                rt=q6.build_runtime(mol,ge,stats.q6_pmf_tables(),mu,sigma,torch.device('cpu'))
                total=float(q6.evaluate(rt,X,ge._q6_a,ge._q6_b)[0]) if rt else 0.0
                ds=(X[:,ge._q6_a]-X[:,ge._q6_b]).norm(dim=-1)[0].numpy() if ge.n_q6 else np.array([])
                phys=rt['rep'].numpy() if rt else np.zeros(len(ds),bool)
                sep=q6.ligand_separation(mol)
                sep14=[]
                for ai,bi in zip(ge._q6_a.tolist(),ge._q6_b.tolist()):
                    a,b=mol.atoms[ai].idx,mol.atoms[bi].idx
                    sep14.append(a!=mol.metal_idx and b!=mol.metal_idx and sep.get(a,{}).get(b)==3)
                sep14=np.asarray(sep14,dtype=bool)
                vals=np.zeros(len(ds))
                if rt and rt.get('rep_any'):
                    vals=q6.rep_energy(torch.as_tensor(ds),rt['rep_A'],rt['rep_b'],rt['rep_rm'],rt['rep_rl']).numpy()
                    vals=np.where(phys,vals,0.0)
                    vals=vals*rt['rep_weight'].numpy()
                    vals=np.where(rt['enabled'].numpy(),vals,0.0)
                pair_rows=[]
                for k,v in enumerate(vals):
                    if phys[k] and v>0.1:
                        a,b=int(ge._q6_a[k]),int(ge._q6_b[k]); aa,bb=mol.atoms[a],mol.atoms[b]
                        pair_sep=sep.get(aa.idx,{}).get(bb.idx)
                        if aa.idx==mol.metal_idx or bb.idx==mol.metal_idx: cause='metal_contact'
                        elif pair_sep==3: cause='ligand_1_4'
                        elif 'Hp' in ge._q6_elem_keys[k] and any(x in kb.Q6_ACCEPTORS for x in ge._q6_elem_keys[k]): cause='polar_H_acceptor'
                        elif pair_sep==4: cause='ligand_1_5_class4'
                        else: cause='class5_or_cross_ligand'
                        pair_rows.append({'i':aa.idx,'j':bb.idx,'elements':[aa.symbol,bb.symbol],
                            'd_A':float(ds[k]),'separation':pair_sep,'cause':cause,'penalty_kT':float(v)})
                p14=float(vals[sep14].sum())
                rows[label]={'total_kT':total,'n_pairs':int(ge.n_q6),'physical_pairs':int(phys.sum()),
                    'physical_penalty_kT':float(vals.sum()),'physical_14_kT':p14,
                    'physical_14_share':p14/float(vals.sum()) if vals.sum()>0 else 0.0,
                    'pairs_over_0p1kT':pair_rows,
                    'pairs_over_1kT':[p for p in pair_rows if p['penalty_kT']>1.0],
                    'disabled_14_pairs':int(np.sum(sep14 & ~rt['enabled'].numpy())) if rt else 0}
            records.append({'code':code,'control':rows['control'],'q6_physical_all':rows['q6_physical_all'],
                'delta_total_kT':rows['q6_physical_all']['total_kT']-rows['control']['total_kT'],
                'delta_physical_kT':rows['q6_physical_all']['physical_penalty_kT']-rows['control']['physical_penalty_kT']})
        except Exception as e:
            failures.append({'code':code,'error':f'{type(e).__name__}: {e}'[:400]})
        if ix%10==0: print(f'{ix}/{len(codes)} completed; failures={len(failures)}',flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(''.join(json.dumps(x)+'\n' for x in records))
    (OUT.parent/'q6_crystal_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps({'n_panel':len(codes),'n_ok':len(records),'failures':failures[:20],'out':str(OUT)},indent=2))
if __name__=='__main__': main()
