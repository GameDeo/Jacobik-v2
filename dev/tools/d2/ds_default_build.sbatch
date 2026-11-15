#!/bin/bash
# Fold-in check (CPU): build the production DB from raw tmQM/lnQM with DEFAULT settings only -- no --profile, no --flags, no --obs-dir, no JACOBIK_* profile
# variables -- and confirm it is the folded-in DB (constraints_2026-11-12.json) entry for entry. Proves the production fit carries the DS methodology by default.
#   1 extract + merge + fit (all stages, default `production` profile, 128 workers)   2 --check-reproduction against the folded-in DB   3 meta check:
#   profile 'production', fingerprint d5e2f4113497, no overrides, differs_from_profile empty, observation_identity observation.1, ancestor_sampling hash_priority.1.
# Needs the owner's per-job go; QOS debug named by the owner 2026-11-12. One extraction + one fit (no second arm): ~15-20 min, cap 30 min, <= 0.5 node-hours (1 CPU node).
# fit.build_db also writes the small lossless constraints.npz and fails the build if it differs from the JSON; the npz is the artifact to adopt as production.
# DS_CODE: clean snapshot of the foldin/ds tip (COMMIT file). DS_OUT: fresh scratch directory. TARGET_DB: the folded-in DB.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-default-build
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
: "${DS_CODE:?set DS_CODE to the committed snapshot}"
: "${DS_OUT:?set DS_OUT to a fresh scratch directory}"
DATA="$JK_SCR/data"; WORKERS="${WORKERS:-128}"
TARGET_DB="${TARGET_DB:-$DATA/db/constraints_2026-11-12.json}"
[ -f "$DS_CODE/COMMIT" ] || { echo "no snapshot at $DS_CODE" >&2; exit 2; }
[ ! -e "$DS_OUT" ] || { echo "output exists: $DS_OUT" >&2; exit 2; }
for f in "$DATA/ref/tmQM" "$DATA/ref/lnQM" "$TARGET_DB" "$DS_CODE/research/bench/benchmarks/locked_exclusions.json"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
FRAME="$DATA/refit/frame_v1/frame.json"
# the whole point: nothing may select a profile or a flag from the environment
unset JACOBIK_PROFILE JACOBIK_PROFILE_OVERRIDES JACOBIK_DB
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$DS_OUT/tmp"; cd "$DS_CODE"
LOCKED="$DS_CODE/research/bench/benchmarks/locked_exclusions.json"
N="$HOME/scripts/jacobik/notify.py"; notify() { "$JK_PY" "$N" "$@" >/dev/null 2>&1 || true; }
echo "commit $(cat COMMIT)  node $(hostname)  workers $WORKERS  start $(date -Is)" | tee "$DS_OUT/job.txt"
notify msg "job $SLURM_JOB_ID node $(hostname) qos ${SLURM_JOB_QOS:-?}: default-settings DB build start (no flags, full extraction + fit, reproduction check vs folded-in DB), cap 30 min"
( while sleep 600; do notify msg "job $SLURM_JOB_ID default build running: $(tail -n 2 "$DS_OUT/job.txt" | tr '\n' ' ') load $(cut -d' ' -f1 /proc/loadavg)"; done ) & watcher=$!
TMPDIR="$DS_OUT/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db --workers "$WORKERS" \
    --ref-dir "$DATA/ref" --out-dir "$DS_OUT/DB" --work-dir "$DS_OUT/DB/work" \
    --locked-path "$LOCKED" --fit-frame-path "$FRAME" --target-db "$TARGET_DB" --check-reproduction > "$DS_OUT/build.log" 2>&1
rc=$?; rm -rf "$DS_OUT/tmp"; echo "build exit $rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
# meta check: a mismatch is reported, not tuned
"$JK_PY" - "$DS_OUT/DB" <<'PY' 2>&1 | tee -a "$DS_OUT/job.txt"
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); m = json.loads((d / "constraints.meta.json").read_text())
want = {"profile": "production", "fingerprint": "d5e2f4113497", "profile_overrides": {}, "differs_from_profile": {},
        "observation_identity": "observation.1", "ancestor_sampling": "hash_priority.1"}
bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
print("meta check:", "OK" if not bad else f"MISMATCH {bad}", "| code_commit", m.get("code_commit"), "| entries", m.get("entries"))
sys.exit(1 if bad else 0)
PY
mr=${PIPESTATUS[0]}; [ $mr = 0 ] || rc=1
kill "$watcher" 2>/dev/null
( cd "$DS_OUT/DB" && sha256sum constraints.json constraints.npz constraints.meta.json > "$DS_OUT/DB.sha256" ) 2>/dev/null
echo "done rc=$rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
notify file "$DS_OUT/job.txt" "$DS_OUT/build.log" --text "job $SLURM_JOB_ID default build end rc=$rc"
exit "$rc"
