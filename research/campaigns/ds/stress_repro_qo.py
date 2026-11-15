import json, time
from pathlib import Path
import numpy as np, pandas as pd
MAIN=Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full"); EV=MAIN/"research/data/audit/2026-11-10_DS"
from fit.pipeline import confidence as stress
from fit.sanity import locked_codes, filter_fit_frame
census=json.loads((EV/"fit_inputs/charge_os_census.json").read_text())
bad={r["csd_code"] for r in census if not r["charge_plausible"]}
locked=locked_codes(MAIN/"research/bench/benchmarks/locked_exclusions.json")
raw=pd.concat([filter_fit_frame(pd.read_parquet(EV/f"fit_inputs/d2_obs/observations_QO_{s}.parquet"),charge_bad=bad,locked=locked,os_sanity=True,exclude_locked=True) for s in ("X1","X2","X3","LNf")],ignore_index=True)
d2=json.loads((MAIN/"research/data/d2/db_d2/constraints_2026-11-09.json").read_text())
for key in [(2,4),(1,4,"N_amidinate")]:
    k=f"QO:{key!r}"
    if k not in d2: print("absent",k); continue
    t=time.time()
    try: f=stress.refit_qo(raw,key)
    except Exception as e: print(k,"ERR",e); continue
    ref=d2[k]; ok=ref["n"]==f["n"] and all(np.allclose(ref[x],f[x],atol=1e-8,rtol=1e-6) for x in ("means","covs","weights"))
    print(k,"n",f["n"],ref["n"],"match",ok,round(time.time()-t,1))
