import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
MAIN=Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
EV=MAIN/"research/data/audit/2026-11-10_DS"
from fit.pipeline import confidence as stress
from fit.sanity import locked_codes, filter_fit_frame
from fit import q1
d2=json.loads((MAIN/"research/data/d2/db_d2/constraints_2026-11-09.json").read_text())
ents={k[3:]:v for k,v in d2.items() if k.startswith("Q1:")}
adapted=[(k,v) for k,v in ents.items() if v["src"]=="adapted" and 20<=v["n"]<=200]
print(len(ents), sum(v["src"]=="adapted" for v in ents.values()), len(adapted))
census=json.loads((EV/"fit_inputs/charge_os_census.json").read_text())
charge_bad={r["csd_code"] for r in census if not r["charge_plausible"]}
locked=locked_codes(MAIN/"research/bench/benchmarks/locked_exclusions.json")
t=time.time()
parts=[]
for sp in ("X1","X2","X3","LNf"):
    raw=pd.read_parquet(EV/f"fit_inputs/d2_obs/observations_Q1_{sp}.parquet")
    parts.append(filter_fit_frame(raw,charge_bad=charge_bad,locked=locked,os_sanity=True,exclude_locked=True))
raw=pd.concat(parts,ignore_index=True); print("rows",len(raw),time.time()-t)
t=time.time(); exp=stress.expand_rows(raw,"Q1"); print("expanded",len(exp),time.time()-t)
from fit import pooling as PL
pf=PL.PooledFitter("Q1",exp,n_indep=q1.N_INDEP)
rng=np.random.default_rng(0)
pick=[adapted[i][0] for i in rng.choice(len(adapted),3,replace=False)]
luq="('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"
for k in pick+[luq]:
    t=time.time(); e=stress.refit_closure(pf,k)
    ref=ents[k]
    d=max(abs(np.array(e["means"])-np.array(ref["means"])).max() if len(e["means"])==len(ref["means"]) else 9, 0)
    print(k[:70], e["src"], ref["src"], "chain",len(stress.closure_keys(pf,k,"%s"%q1.DEFAULT_VARIANT)), "k",e["k"],ref["k"],"maxdiff means",d, round(time.time()-t,1))
