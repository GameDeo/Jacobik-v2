#!/bin/bash
# DS job J1 (CPU): (1) rebuild the candidate DB DF from the retained E_assign tables at the gate branch tip, so its .meta.json records the code commit,
#   and check it against the Step 4 D-final key by key; (2) the geometry_0.02A stress trials (refit stage) for LUQVOK Q1 and LIPDEU QO:(2, 4).
#   (1) one fit, ~10 min (D-final took 10.6 min while sharing the node with two other fits). (2) 96 + 128 trials, started together with (1). Estimate ~14 min, cap 45 min, <= 0.4 node-hours.
# Needs the owner's per-job go (QOS regular). DS_CODE: clean snapshot of the gate branch (COMMIT file); PIPE: the Step 4 pipeline output; DS_OUT: fresh directory.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -J jk-ds-final
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
: "${DS_CODE:?}"; : "${PIPE:?}"; : "${DS_OUT:?}"
DATA="$JK_SCR/data"; WORKERS="${WORKERS:-96}"   # rebuild workers; 96 + 32 + 16 geometry workers share the 128 cores
[ -f "$DS_CODE/COMMIT" ] || { echo "no snapshot at $DS_CODE" >&2; exit 2; }
[ ! -e "$DS_OUT" ] || { echo "output exists: $DS_OUT" >&2; exit 2; }
for f in "$PIPE/E_assign/work/obs" "$PIPE/E_off/work/obs" "$PIPE/D-final/constraints.json" "$PIPE/D-final/charge_os_census.json" "$PIPE/E_assign.obs.sha256" \
         "$DATA/ref/tmQM" "$DATA/ref/lnQM" "$DS_CODE/research/bench/benchmarks/locked_exclusions.json"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }; done
FRAME="$DATA/refit/frame_v1/frame.json"
( cd "$PIPE/E_assign/work" && sha256sum -c --quiet "$PIPE/E_assign.obs.sha256" ) || { echo "retained tables differ from their recorded hashes" >&2; exit 2; }
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench:$DS_CODE/research/campaigns/ds" JK_MAIN="$DS_CODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$DS_OUT"; cd "$DS_CODE"
LOCKED="$DS_CODE/research/bench/benchmarks/locked_exclusions.json"
N="$HOME/scripts/jacobik/notify.py"; notify() { "$JK_PY" "$N" "$@" >/dev/null 2>&1 || true; }
echo "commit $(cat COMMIT)  node $(hostname)  start $(date -Is)" | tee "$DS_OUT/job.txt"
echo '{"donor_identity_reconcile": true, "donor_assignment": true, "stable_ancestor_sampling": true, "structure_sanity": true}' > "$DS_OUT/DF.flags.json"
notify msg "job $SLURM_JOB_ID node $(hostname) qos regular: DS final start (rebuild DF + reproduction check, then geometry stress), cap 45 min"
( while sleep 600; do notify msg "job $SLURM_JOB_ID DS final running: $(tail -n 2 "$DS_OUT/job.txt" | tr '\n' ' ') load $(cut -d' ' -f1 /proc/loadavg)"; done ) & watcher=$!
rc=0
# The geometry refits need only the Step 4 pipeline output, not the rebuild, and the rebuild leaves most cores idle for ~3 of its ~10 minutes (expansion,
# assemble and reproduction are nearly serial), so everything starts together: ~55,000 core-seconds on 128 cores instead of ~22 minutes in sequence.
S=research/campaigns/ds/stress_run.py
/usr/bin/time -v "$JK_PY" $S refit "$PIPE" "$DS_OUT/luqvok_q1.geometry.entries.jsonl" --case LUQVOK --quantity Q1 \
  --key "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')" --workers 32 --ref-dir "$DATA/ref" --scenarios geometry_0.02A > "$DS_OUT/luqvok_q1.geometry.log" 2>&1 &
p1=$!
/usr/bin/time -v "$JK_PY" $S refit "$PIPE" "$DS_OUT/lipdeu_qo.geometry.entries.jsonl" --case LIPDEU --quantity QO --key "[2, 4]" --workers 16 --ref-dir "$DATA/ref" \
  --scenarios geometry_0.02A > "$DS_OUT/lipdeu_qo.geometry.log" 2>&1 &
p2=$!
# rebuild DF from the retained tables; --check-reproduction compares it with the pipeline D-final (128 workers: the node's cores)
mkdir -p "$DS_OUT/DF/tmp"
TMPDIR="$DS_OUT/DF/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db --profile d2 --flags "$DS_OUT/DF.flags.json" --workers "$WORKERS" \
    --ref-dir "$DATA/ref" --out-dir "$DS_OUT/DF" --obs-dir "$PIPE/E_assign/work/obs" --work-dir "$DS_OUT/DF/work" \
    --locked-path "$LOCKED" --fit-frame-path "$FRAME" --target-db "$PIPE/D-final/constraints.json" --check-reproduction > "$DS_OUT/DF.log" 2>&1
r=$?; rm -rf "$DS_OUT/DF/tmp"; echo "DF rebuild exit $r $(date -Is)" | tee -a "$DS_OUT/job.txt"; [ $r = 0 ] || rc=1
wait $p1 || { rc=1; echo "LUQVOK geometry refit failed" | tee -a "$DS_OUT/job.txt"; }
wait $p2 || { rc=1; echo "LIPDEU geometry refit failed" | tee -a "$DS_OUT/job.txt"; }
kill "$watcher" 2>/dev/null
( cd "$DS_OUT/DF" && sha256sum constraints.json constraints.meta.json > "$DS_OUT/DF.sha256" )
echo "done rc=$rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
notify file "$DS_OUT/job.txt" "$DS_OUT/DF.log" "$DS_OUT/luqvok_q1.geometry.log" "$DS_OUT/lipdeu_qo.geometry.log" --text "job $SLURM_JOB_ID DS final end rc=$rc"
exit "$rc"
