"""Exactness and cost of the chain-restricted stable refit: same entry as the full expansion, how much faster."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from fit.pipeline import confidence as registry
stress = stress_driver = registry
from fit import pooling as PL
from fit.sanity import locked_codes, filter_fit_frame
KEY = "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"
P = Path(sys.argv[1]); arm = "D-final"
census = json.loads((P / arm / "charge_os_census.json").read_text())
bad = {r["csd_code"] for r in census if not r["charge_plausible"] or not r.get("structure_plausible", True)}
locked = locked_codes(MAIN / "research/bench/benchmarks/locked_exclusions.json")
raw = pd.concat([filter_fit_frame(pd.read_parquet(P / "E_assign" / f"work/obs/observations_Q1_{s}.parquet"), charge_bad=bad, locked=locked,
                                  os_sanity=True, exclude_locked=True) for s in ("X1", "X2", "X3", "LNf")], ignore_index=True)
t = time.time()
expanded = stress.expand_rows(raw, "Q1"); n_indep, _, _ = stress._spec("Q1")
pf = PL.PooledFitter("Q1", expanded, n_indep=n_indep)
chain = stress.closure_keys(pf, KEY, None)
print("full expansion + chain %.0fs; chain keys %d; rows %d -> feeding rows %d" % (time.time() - t, len(chain), len(raw), int(stress.feeds_closure(raw, "Q1", chain).sum())), flush=True)
support = raw.loc[raw.key == KEY, "csd_code"].astype(str).tolist()
reg = registry.register({"quantity": "Q1", "resolved_key": KEY, "parent_code": "LUQVOK"}, support, {"base": "0" * 64}, scenarios=("parent_removal",), replicates=2)
for trial in [x for x in reg["trials"] if x["affected_count"] in (1, 5)][:2]:
    t0 = time.time(); a = stress_driver.refit_trial(raw, "Q1", KEY, trial, "parent_removal", stable=True, chain=chain); t1 = time.time()
    b = stress_driver.refit_trial(raw, "Q1", KEY, trial, "parent_removal", stable=True); t2 = time.time()
    same = all(np.allclose(np.asarray(a[k], float), np.asarray(b[k], float), rtol=0, atol=0) for k in ("means", "vars", "weights"))
    print("n=%d restricted %.1fs full %.1fs identical=%s" % (trial["affected_count"], t1 - t0, t2 - t1, same), np.round(a["means"], 4), flush=True)
