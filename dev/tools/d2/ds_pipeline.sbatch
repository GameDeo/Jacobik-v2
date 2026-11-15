#!/bin/bash
# DS step 4: staged extraction -> four fits (D0, D-data, D-sample, D-both) in ONE job. Supersedes ds_data_build.sbatch.
#   round 1  extract E_off (donor_identity_reconcile off) and E_assign (reconcile + donor_assignment), stable IDs on, stop after merge
#   round 2  D0 (E_off, legacy sampler, --check-reproduction against the recorded D2)  ||  D-data (E_assign, legacy sampler)
#   round 3  D-sample (E_off, stable ancestor sampling)  ||  D-both (E_assign, stable ancestor sampling)  ||  D-final (D-both + structure_sanity: the candidate DB)
# Fits read the retained extraction tables (--obs-dir): sources, models, charge policy and locked exclusions are identical across arms.
# Needs the owner's per-job go for QOS regular. Estimate: round 1 ~4 min, rounds 2 and 3 ~5 min each => ~15 min, cap 35 min, <= 0.6 node-hours.
# DS_CODE: clean snapshot of the committed branch (COMMIT file). DS_OUT: fresh scratch directory. TARGET_DB: recorded D2 for the D0 check.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:35:00
#SBATCH -J jk-ds-pipeline
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
: "${DS_CODE:?set DS_CODE to the committed snapshot}"
: "${DS_OUT:?set DS_OUT to a fresh scratch directory}"
DATA="$JK_SCR/data"; WORKERS="${WORKERS:-96}"
TARGET_DB="${TARGET_DB:-$DATA/db_d2/constraints_2026-11-09.json}"
[ -f "$DS_CODE/COMMIT" ] || { echo "no snapshot at $DS_CODE" >&2; exit 2; }
[ ! -e "$DS_OUT" ] || { echo "output exists: $DS_OUT" >&2; exit 2; }
for f in "$DATA/ref/tmQM" "$DATA/ref/lnQM" "$TARGET_DB" "$DS_CODE/research/bench/benchmarks/locked_exclusions.json"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
FRAME="$DATA/refit/frame_v1/frame.json"
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$DS_OUT"; cd "$DS_CODE"
LOCKED="$DS_CODE/research/bench/benchmarks/locked_exclusions.json"
N="$HOME/scripts/jacobik/notify.py"; notify() { "$JK_PY" "$N" "$@" >/dev/null 2>&1 || true; }
echo "commit $(cat COMMIT)  node $(hostname)  workers $WORKERS  start $(date -Is)" | tee "$DS_OUT/job.txt"
flags() {  # name reconcile assignment stable [structure_sanity]
  echo "{\"donor_identity_reconcile\": $2, \"donor_assignment\": $3, \"stable_ancestor_sampling\": $4, \"structure_sanity\": ${5:-false}}" > "$DS_OUT/$1.flags.json"; }
flags E_off false false true;  flags E_assign true true true
flags D0 false false false;    flags D-data true true false
flags D-sample false false true; flags D-both true true true; flags D-final true true true true
extract() {
  local n=$1; mkdir -p "$DS_OUT/$n/tmp"
  TMPDIR="$DS_OUT/$n/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db --profile d2 --flags "$DS_OUT/$n.flags.json" --workers "$WORKERS" \
      --stop-after merge --ref-dir "$DATA/ref" --out-dir "$DS_OUT/$n" --work-dir "$DS_OUT/$n/work" \
      --locked-path "$LOCKED" --fit-frame-path "$FRAME" > "$DS_OUT/$n.log" 2>&1
  local rc=$?; rm -rf "$DS_OUT/$n/tmp"; echo "$n exit $rc $(date -Is)" | tee -a "$DS_OUT/job.txt"; return $rc
}
fit() {  # name obs-dir-source [extra args]
  local n=$1 src=$2; shift 2; mkdir -p "$DS_OUT/$n/tmp"
  TMPDIR="$DS_OUT/$n/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db --profile d2 --flags "$DS_OUT/$n.flags.json" --workers "$WORKERS" \
      --ref-dir "$DATA/ref" --out-dir "$DS_OUT/$n" --obs-dir "$DS_OUT/$src/work/obs" --work-dir "$DS_OUT/$n/work" \
      --locked-path "$LOCKED" --fit-frame-path "$FRAME" --target-db "$TARGET_DB" "$@" > "$DS_OUT/$n.log" 2>&1
  local rc=$?; rm -rf "$DS_OUT/$n/tmp"; echo "$n exit $rc $(date -Is)" | tee -a "$DS_OUT/job.txt"; return $rc
}
round() { local rc=0 pids=(); for spec in "$@"; do eval "$spec" & pids+=($!); done; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc; }
notify msg "job $SLURM_JOB_ID node $(hostname) qos regular: DS pipeline start (extract E_off,E_assign -> fit D0,D-data -> D-sample,D-both), cap 35 min"
( while sleep 600; do notify msg "job $SLURM_JOB_ID DS pipeline running: $(tail -n 3 "$DS_OUT/job.txt" | tr '\n' ' ') load $(cut -d' ' -f1 /proc/loadavg)"; done ) & watcher=$!
rc=0
round "extract E_off" "extract E_assign" || { echo "extraction failed" | tee -a "$DS_OUT/job.txt"; rc=1; }
if [ $rc = 0 ]; then
  # D0 failing its reproduction check is a result to report, not a reason to stop the other arms
  round "fit D0 E_off --check-reproduction" "fit D-data E_assign" || echo "round 2 had a failure (see logs; D0 reproduction mismatch is reported, not tuned)" | tee -a "$DS_OUT/job.txt"
  round "fit D-sample E_off" "fit D-both E_assign" "fit D-final E_assign" || echo "round 3 had a failure" | tee -a "$DS_OUT/job.txt"
fi
kill "$watcher" 2>/dev/null
for n in E_off E_assign; do [ -d "$DS_OUT/$n/work/obs" ] && (cd "$DS_OUT/$n/work" && sha256sum obs/* > "$DS_OUT/$n.obs.sha256"); done
echo "done rc=$rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
notify file "$DS_OUT/job.txt" "$DS_OUT/D0.log" --text "job $SLURM_JOB_ID DS pipeline end rc=$rc"
exit "$rc"
