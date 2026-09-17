#!/usr/bin/env python3
"""
Safely merge the refit Q2/Q3 GMM-on-cos(angle) entries into the live
constraints.json.

fit_all()'s `database` dict starts EMPTY (not preloaded from the existing
constraints.json) -- so the refit's temp checkpoint file only contains
Q2/Q3 entries. This script:
  1. Backs up the live constraints.json (timestamped copy, never overwritten).
  2. Loads the live DB and the refit checkpoint.
  3. Removes every old "Q2:"/"Q3:" entry from the live DB.
  4. Inserts every entry from the refit checkpoint.
  5. Writes the merged DB back to the live path.
  6. Verifies: no Q2/Q3 entry has a "kappas" field, every Q2/Q3 entry has
     type == "gmm_cos", and every OTHER quantity's entries are byte-for-byte
     unchanged from before the merge.
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../Jacobik v3
LIVE_PATH = ROOT / "src" / "jacobik" / "constraints.json"
REFIT_PATH = ROOT / "_patch_scripts" / "constraints_q2q3_refit.json"
BACKUP_PATH = ROOT / "src" / "jacobik" / f"constraints.json.bak_pre_q2q3_refit_{time.strftime('%Y%m%dT%H%M%S')}"

assert REFIT_PATH.exists(), f"Refit checkpoint not found: {REFIT_PATH}"

with open(LIVE_PATH, "r", encoding="utf-8") as f:
    live = json.load(f)
print(f"Live DB: {len(live)} total entries")

with open(REFIT_PATH, "r", encoding="utf-8") as f:
    refit = json.load(f)
print(f"Refit checkpoint: {len(refit)} entries")

assert all(k.startswith("Q2:") or k.startswith("Q3:") for k in refit), (
    "Refit checkpoint contains keys outside Q2/Q3 -- refusing to merge (would silently "
    "clobber other quantities if something went wrong upstream)."
)

old_q2 = {k for k in live if k.startswith("Q2:")}
old_q3 = {k for k in live if k.startswith("Q3:")}
new_q2 = {k for k in refit if k.startswith("Q2:")}
new_q3 = {k for k in refit if k.startswith("Q3:")}
print(f"Live: Q2={len(old_q2)} Q3={len(old_q3)}")
print(f"Refit: Q2={len(new_q2)} Q3={len(new_q3)}")

# Snapshot every non-Q2/Q3 entry for the post-merge invariance check.
other_before = {k: v for k, v in live.items() if not (k.startswith("Q2:") or k.startswith("Q3:"))}

# ---- Backup -------------------------------------------------------------
shutil.copy2(LIVE_PATH, BACKUP_PATH)
print(f"Backed up live DB to: {BACKUP_PATH}")

# ---- Merge ----------------------------------------------------------------
merged = dict(live)
for k in list(merged.keys()):
    if k.startswith("Q2:") or k.startswith("Q3:"):
        del merged[k]
merged.update(refit)

print(f"Merged DB: {len(merged)} total entries")
print(f"  (was {len(live)}, removed {len(old_q2) + len(old_q3)} old Q2/Q3, added {len(refit)} new)")

# ---- Write ------------------------------------------------------------------
with open(LIVE_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, separators=(",", ":"))
print(f"Wrote merged DB to {LIVE_PATH}")

# ---- Verify -----------------------------------------------------------------
with open(LIVE_PATH, "r", encoding="utf-8") as f:
    verify = json.load(f)

n_bad_type = 0
n_has_kappas = 0
for k, v in verify.items():
    if k.startswith("Q2:") or k.startswith("Q3:"):
        if v.get("type") != "gmm_cos":
            n_bad_type += 1
            print(f"  BAD TYPE: {k} -> {v.get('type')}")
        if "kappas" in v:
            n_has_kappas += 1
            print(f"  STILL HAS KAPPAS: {k}")

other_after = {k: v for k, v in verify.items() if not (k.startswith("Q2:") or k.startswith("Q3:"))}
other_unchanged = other_before == other_after

print()
print("=== VERIFICATION ===")
print(f"Total entries: {len(verify)}")
print(f"Q2/Q3 entries with type != 'gmm_cos': {n_bad_type}")
print(f"Q2/Q3 entries still carrying a 'kappas' field: {n_has_kappas}")
print(f"All non-Q2/Q3 entries byte-identical to before merge: {other_unchanged}")
if not other_unchanged:
    only_before = set(other_before) - set(other_after)
    only_after = set(other_after) - set(other_before)
    changed = {k for k in other_before.keys() & other_after.keys() if other_before[k] != other_after[k]}
    print(f"  missing after merge: {len(only_before)}")
    print(f"  new/unexpected after merge: {len(only_after)}")
    print(f"  changed value: {len(changed)}")

ok = (n_bad_type == 0) and (n_has_kappas == 0) and other_unchanged and (len(verify) == len(merged))
print()
print("MERGE_OK" if ok else "MERGE_FAILED")
