#!/usr/bin/env python3
"""
Resumable Q2/Q3 GMM-on-cos(angle) refit driver.

Each remote-shell call is a fresh process namespace with a hard wall-clock
cap, so a single long-running fit_all() call (14651 Q2 keys + 2956 Q3 keys,
only 2 CPUs available) can't be background-and-poll'd the normal way --
child processes started with nohup/disown do not survive past the call that
spawned them here. Instead this script fits keys ONE AT A TIME (serially;
no process pool -- avoids any pool/IPC complications across repeated short
calls), checkpointing the accumulated results to CHECKPOINT_PATH after
every key and stopping cleanly once TIME_BUDGET_S has elapsed. Re-running
this script picks up exactly where the previous invocation left off (already
-fitted keys are skipped). Writes DONE_MARKER when every key has been
processed.

Does NOT touch the live constraints.json -- fit_all()'s pattern of starting
from an empty `database` applies here too; merge_constraints.py does the
actual safe merge into the live file afterwards.
"""
import importlib.util
import json
import time
from pathlib import Path

TIME_BUDGET_S = 150.0  # soft budget per invocation; leaves margin under the ~180s hard cap

ROOT = Path(__file__).resolve().parents[1]  # .../Jacobik v3
FITTER_PATH = ROOT / "src" / "jacobik" / "stats" / "fitter.py"
EXPANDED_DIR = ROOT / "data" / "out" / "expanded"
CHECKPOINT_PATH = ROOT / "_patch_scripts" / "constraints_q2q3_refit.json"
SKIPPED_PATH = ROOT / "_patch_scripts" / "refit_skipped.json"
DONE_MARKER = ROOT / "_patch_scripts" / "refit_DONE"

spec = importlib.util.spec_from_file_location("jacobik_fitter_standalone", FITTER_PATH)
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)

assert fitter._DIST_TYPE["Q2"] == "gmm_cos", "fitter.py patch did not apply -- Q2 still vmm"
assert fitter._DIST_TYPE["Q3"] == "gmm_cos", "fitter.py patch did not apply -- Q3 still vmm"

if DONE_MARKER.exists():
    print("ALL_DONE (marker already present, nothing to do)")
    raise SystemExit(0)

t_start = time.perf_counter()

# ---- Load checkpoint (accumulated results so far) --------------------------
if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        database = json.load(f)
    print(f"Resuming from checkpoint: {len(database)} keys already fitted.")
else:
    database = {}
    print("No checkpoint found -- starting fresh.")

if SKIPPED_PATH.exists():
    with open(SKIPPED_PATH, "r", encoding="utf-8") as f:
        skipped_keys = set(json.load(f))
    print(f"{len(skipped_keys)} keys previously skipped (fit returned None) -- will not retry.")
else:
    skipped_keys = set()

# ---- Load observations for both quantities ---------------------------------
t0 = time.perf_counter()
all_items = []
for q in ("Q2", "Q3"):
    obs = fitter._load_observations(EXPANDED_DIR, q, fitter.N_MIN)
    for key_str, values in obs.items():
        all_items.append((q, key_str, values))
print(f"Loaded {len(all_items)} total (Q2+Q3) keys in {time.perf_counter() - t0:.1f}s")

remaining = [
    (q, k, v) for (q, k, v) in all_items if f"{q}:{k}" not in database and f"{q}:{k}" not in skipped_keys
]
n_already_handled = len(all_items) - len(remaining)
print(f"{len(remaining)} keys remaining to fit ({n_already_handled} already done or skipped).")

# ---- Fit one key at a time, checkpointing periodically ----------------------
n_fitted_this_run = 0
n_skipped_this_run = 0
CHECKPOINT_EVERY = 200

for i, (quantity, key_str, values) in enumerate(remaining):
    try:
        params = fitter.fit_gmm_cosangle(values, fitter._K_MAX[quantity])
    except Exception as exc:
        print(f"  FIT FAILED {quantity} {key_str!r}: {exc}")
        params = None

    if params is not None:
        database[f"{quantity}:{key_str}"] = params
        n_fitted_this_run += 1
    else:
        skipped_keys.add(f"{quantity}:{key_str}")
        n_skipped_this_run += 1

    elapsed = time.perf_counter() - t_start
    if (i + 1) % CHECKPOINT_EVERY == 0:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(database, f)
        with open(SKIPPED_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(skipped_keys), f)
        print(
            f"  ... {i + 1}/{len(remaining)} this run  "
            f"(fitted={n_fitted_this_run} skipped={n_skipped_this_run})  "
            f"elapsed={elapsed:.0f}s  total_db={len(database)}"
        )

    if elapsed > TIME_BUDGET_S:
        print(f"Time budget ({TIME_BUDGET_S}s) reached -- checkpointing and stopping.")
        break
else:
    # Completed the whole `remaining` list without hitting the time budget.
    pass

# ---- Final checkpoint write --------------------------------------------------
with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
    json.dump(database, f)
with open(SKIPPED_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted(skipped_keys), f)

n_handled_total = len(database) + len(skipped_keys)
n_expected_total = len(all_items)
print(
    f"Checkpoint written: {len(database)} fitted + {len(skipped_keys)} skipped = "
    f"{n_handled_total}/{n_expected_total} keys handled  "
    f"(this run: fitted={n_fitted_this_run} skipped={n_skipped_this_run}, "
    f"{time.perf_counter() - t_start:.1f}s)"
)

if n_handled_total >= n_expected_total:
    DONE_MARKER.write_text("done\n")
    print("ALL_DONE")
else:
    print(f"NOT_DONE_YET  ({n_expected_total - n_handled_total} keys remaining)")
