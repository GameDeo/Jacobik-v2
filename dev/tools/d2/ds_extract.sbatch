#!/bin/bash
# DS step 3: fresh extraction only (ledger + persistent observation IDs), then STOP for source review. No fits.
# Two extractions run concurrently on one CPU node, 96 workers each (192 of 256 hardware threads):
#   E_off  donor_identity_reconcile=false, stable IDs on   -> feeds D0 (legacy sampler) and D-sample
#   E_on   donor_identity_reconcile=true,  stable IDs on   -> feeds D-data (legacy sampler) and D-both
# Needs the owner's per-job go for QOS regular. Estimate 8-12 min wall, cap 25 min, <= 0.42 node-hours.
# DS_CODE: clean snapshot of the committed branch (COMMIT file). DS_OUT: fresh scratch directory.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -J jk-ds-extract
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
: "${DS_CODE:?set DS_CODE to the committed snapshot}"
: "${DS_OUT:?set DS_OUT to a fresh scratch directory}"
DATA="$JK_SCR/data"; WORKERS="${WORKERS:-96}"
[ -f "$DS_CODE/COMMIT" ] || { echo "no snapshot at $DS_CODE" >&2; exit 2; }
[ ! -e "$DS_OUT" ] || { echo "output exists: $DS_OUT" >&2; exit 2; }
for f in "$DATA/ref/tmQM" "$DATA/ref/lnQM" "$DS_CODE/research/bench/benchmarks/locked_exclusions.json"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
FRAME="$DATA/refit/frame_v1/frame.json"
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$DS_OUT"; cd "$DS_CODE"
echo "commit $(cat COMMIT)  node $(hostname)  workers/extraction $WORKERS  start $(date -Is)" | tee "$DS_OUT/job.txt"
echo '{"donor_identity_reconcile": false, "stable_ancestor_sampling": true}' > "$DS_OUT/E_off.flags.json"
echo '{"donor_identity_reconcile": true,  "stable_ancestor_sampling": true}' > "$DS_OUT/E_on.flags.json"
extract() {
  local name=$1; mkdir -p "$DS_OUT/$name/tmp"
  TMPDIR="$DS_OUT/$name/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db \
      --profile d2 --flags "$DS_OUT/$name.flags.json" --workers "$WORKERS" --stop-after merge \
      --ref-dir "$DATA/ref" --out-dir "$DS_OUT/$name" --work-dir "$DS_OUT/$name/work" \
      --locked-path "$DS_CODE/research/bench/benchmarks/locked_exclusions.json" \
      --fit-frame-path "$FRAME" > "$DS_OUT/$name.log" 2>&1
  local rc=$?; rm -rf "$DS_OUT/$name/tmp"; echo "$name exit $rc $(date -Is)" | tee -a "$DS_OUT/job.txt"; return $rc
}
N="$HOME/scripts/jacobik/notify.py"   # plain text; a silent no-op without a webhook
notify() { "$JK_PY" "$N" "$@" >/dev/null 2>&1 || true; }
notify msg "job $SLURM_JOB_ID node $(hostname) qos regular: extraction start (${ONLY:-E_off E_on}), $WORKERS workers each, cap 25 min"
( while sleep 600; do notify msg "job $SLURM_JOB_ID extraction running: $(ls "$DS_OUT"/E_*/work/shards 2>/dev/null | wc -l) shard files, load $(cut -d' ' -f1 /proc/loadavg)"; done ) & watcher=$!
# ONLY=E_on re-extracts just the reconcile-on tables (E_off is unaffected by changes to the reconcile rule)
pids=()
for n in ${ONLY:-E_off E_on}; do extract "$n" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
kill "$watcher" 2>/dev/null
# one hashed manifest of the merged tables and ledgers (inode quota: the work dirs stay as they are, tar the shards)
for n in ${ONLY:-E_off E_on}; do (cd "$DS_OUT/$n/work" && sha256sum obs/* > "$DS_OUT/$n.obs.sha256"); done
echo "done rc=$rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
notify file "$DS_OUT/job.txt" "$DS_OUT/E_off.log" "$DS_OUT/E_on.log" --text "job $SLURM_JOB_ID extraction end rc=$rc"
exit "$rc"
