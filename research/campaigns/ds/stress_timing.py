"""Cost of one stress trial on a retained extraction (stable sampler): LUQVOK Q1 key, refit and fixed-pool scoring under the base DB.

    python stress_timing.py PIPELINE_DIR BASE_ARM(D-final)
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full")
from fit.pipeline import confidence
registry = stress_driver = confidence
from fit.sanity import locked_codes, filter_fit_frame
from tools.jgen.ds import stress_trial as ST
KEY = "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"
P, arm = Path(sys.argv[1]), sys.argv[2]
ext = "E_assign" if arm in ("D-both", "D-final", "D-data") else "E_off"
t = time.time()
census = json.loads((P / arm / "charge_os_census.json").read_text())
bad = {r["csd_code"] for r in census if not r["charge_plausible"]}
if arm == "D-final":
    bad |= {r["csd_code"] for r in census if not r.get("structure_plausible", True)}
locked = locked_codes(MAIN / "research/bench/benchmarks/locked_exclusions.json")
raw = pd.concat([filter_fit_frame(pd.read_parquet(P / ext / f"work/obs/observations_Q1_{s}.parquet"), charge_bad=bad, locked=locked,
                                  os_sanity=True, exclude_locked=True) for s in ("X1", "X2", "X3", "LNf")], ignore_index=True)
# observed-error replacement for a base built on the corrected extraction: the same parents' rows from the uncorrected extraction
rep = pd.concat([filter_fit_frame(pd.read_parquet(P / "E_off" / f"work/obs/observations_Q1_{s}.parquet"), charge_bad=bad, locked=locked,
                                  os_sanity=True, exclude_locked=True) for s in ("X1", "X2", "X3", "LNf")], ignore_index=True) if ext == "E_assign" else None
print("raw", len(raw), "load %.1fs" % (time.time() - t), flush=True)
support = raw.loc[raw.key == KEY, "csd_code"].astype(str).tolist()
reg = registry.register({"quantity": "Q1", "resolved_key": KEY, "parent_code": "LUQVOK"}, support, {"base": "0" * 64},
                        scenarios=("parent_removal", "observed_input_error"), replicates=2)
print("support parents", reg["support_count"], "trials", len(reg["trials"]), flush=True)
score_fn, clean, info = ST.prepare(ST.manifest_row(MAIN, "LUQVOK", "D"), resolved_key=eval(KEY), score_db=str(P / arm / "constraints.json"))
print(info, flush=True)
sel = [t_ for t_ in reg["trials"] if t_["scenario"] == "parent_removal"][:2] + [t_ for t_ in reg["trials"] if t_["scenario"] == "observed_input_error"][:2]
for trial in sel:
    t0 = time.time()
    try:
        entry = stress_driver.refit_trial(raw, "Q1", KEY, trial, trial["scenario"], stable=True, replacements=rep)
    except Exception as exc:
        print(trial["scenario"], "refit failed:", type(exc).__name__, str(exc)[:160], flush=True); continue
    t1 = time.time()
    res = confidence.evaluate(clean, score_fn(stress_driver.fitdist_kwargs(entry)))
    print(trial["scenario"], "n=%d" % trial["affected_count"], "means", np.round(entry["means"], 3), "refit %.1fs score %.1fs" % (t1 - t0, time.time() - t1),
          "loss", np.round(res["energy_loss_kj"], 1), "severe", res["severe"], flush=True)
