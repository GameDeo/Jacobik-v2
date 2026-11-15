"""Cost and exactness of one LIPDEU QO (2, 4) stress trial on the corrected extraction."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from fit.pipeline import confidence as registry
stress = stress_driver = registry
from fit.sanity import locked_codes, filter_fit_frame
P = Path(sys.argv[1]); GROUP = (2, 4)
census = json.loads((P / "D-final" / "charge_os_census.json").read_text())
bad = {r["csd_code"] for r in census if not r["charge_plausible"] or not r.get("structure_plausible", True)}
locked = locked_codes(MAIN / "research/bench/benchmarks/locked_exclusions.json")
raw = pd.concat([filter_fit_frame(pd.read_parquet(f), charge_bad=bad, locked=locked, os_sanity=True, exclude_locked=True)
                 for f in sorted((P / "E_assign/work/obs").glob("observations_QO_*.parquet"))], ignore_index=True)
print("QO rows", len(raw), list(raw.columns)[:8], flush=True)
t = time.time(); clean = stress.refit_qo(raw, GROUP); print("clean refit %.1fs means" % (time.time() - t), np.round(clean["means"], 4), "n", clean.get("n"), flush=True)
db = json.loads((P / "D-final/constraints.json").read_text())
ref = db["QO:(2, 4)"] if "QO:(2, 4)" in db else db["entries"]["QO:(2, 4)"]
print("matches D-final entry:", np.allclose(np.sort(np.ravel(ref["means"])), np.sort(np.ravel(clean["means"])), atol=1e-9), flush=True)
