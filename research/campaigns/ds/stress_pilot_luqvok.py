"""Pilot: LUQVOK Q1 resolved key, parent_removal trials through the full driver chain (refit -> fixed-pool scoring -> severity)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
MAIN = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full"); EV = MAIN / "research/data/audit/2026-11-10_DS"
from fit.pipeline import confidence
registry = stress_driver = confidence
from fit.sanity import locked_codes, filter_fit_frame
from tools.jgen.ds import stress_trial as ST
KEY = "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"
CULPRITS = "BOPSAD CEQHIR FIFJIO GORTEP MAKJAM MAKJEQ NATBOZ REYKUF TEHWUY VIHKEE WESZEA".split()
t = time.time()
census = json.loads((EV / "fit_inputs/charge_os_census.json").read_text())
charge_bad = {r["csd_code"] for r in census if not r["charge_plausible"]}
locked = locked_codes(MAIN / "research/bench/benchmarks/locked_exclusions.json")
raw = pd.concat([filter_fit_frame(pd.read_parquet(EV / f"fit_inputs/d2_obs/observations_Q1_{s}.parquet"), charge_bad=charge_bad,
                                  locked=locked, os_sanity=True, exclude_locked=True) for s in ("X1", "X2", "X3", "LNf")], ignore_index=True)
print("raw", len(raw), round(time.time() - t, 1), "s", flush=True)
support = raw.loc[raw.key == KEY, "csd_code"].astype(str).tolist()
reg = registry.register({"quantity": "Q1", "resolved_key": KEY, "parent_code": "LUQVOK"}, support, {"d2_db": "0" * 64},
                        scenarios=("parent_removal",), replicates=2)
print("support parents", reg["support_count"], "trials", len(reg["trials"]), flush=True)
score_fn, clean, info = ST.prepare(ST.manifest_row(MAIN, "LUQVOK", "D"), resolved_key=eval(KEY))
print(info, flush=True)
from jacobik.stats.mixture import structure_id
pick = [t_ for t_ in reg["trials"] if t_["affected_count"] in (1, 4)][:3]
out = []
for trial in pick + [dict(trial_id="culprits", scenario="parent_removal", replicate=-1, affected_count=11,
                           affected_parents=sorted({structure_id(c) for c in CULPRITS}), actual_fraction=11 / reg["support_count"])]:
    t0 = time.time()
    entry = stress_driver.refit_trial(raw, "Q1", KEY, trial, "parent_removal", stable=False)
    t1 = time.time()
    res = confidence.evaluate(clean, score_fn(stress_driver.fitdist_kwargs(entry)))
    print(trial["trial_id"][:8], "n=%d" % trial["affected_count"], "means", np.round(entry["means"], 3), "w", np.round(entry["weights"], 3),
          "| refit %.0fs score %.1fs" % (t1 - t0, time.time() - t1), "| loss kJ", np.round(res["energy_loss_kj"], 1), "severe", res["severe"], flush=True)
    out.append(dict(trial=trial["trial_id"], means=entry["means"], weights=entry["weights"], outcome=res))
Path("research/data/ds_proto").mkdir(parents=True, exist_ok=True)
Path("research/data/ds_proto/stress_pilot_luqvok.json").write_text(json.dumps(out, indent=1, default=str))
