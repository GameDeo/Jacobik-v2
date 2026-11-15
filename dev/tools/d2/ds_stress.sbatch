#!/bin/bash
# DS step 5b pilot: refit stage of the registered stress design for LUQVOK Q1 and LIPDEU QO:(2, 4) on the D-final base (scoring is done afterwards locally).
#   observed_input_error + parent_removal (geometry_0.02A is added by a later job once its executor is verified), 32 replicates, doses 1/1%/2%/5%.
#   Q1: 192 trials x ~100 s; QO: 256 trials x ~30 s; 48 workers each => estimate ~12 min wall, cap 30 min, <= 0.5 node-hours.
# Needs the owner's per-job go (QOS regular). DS_CODE: clean snapshot (COMMIT file); PIPE: the Step 4 output directory; DS_OUT: fresh directory.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-ds-stress
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
: "${DS_CODE:?}"; : "${PIPE:?}"; : "${DS_OUT:?}"
[ -f "$DS_CODE/COMMIT" ] || { echo "no snapshot at $DS_CODE" >&2; exit 2; }
[ ! -e "$DS_OUT" ] || { echo "output exists: $DS_OUT" >&2; exit 2; }
for f in "$PIPE/E_assign/work/obs" "$PIPE/E_off/work/obs" "$PIPE/D-final/constraints.json" "$PIPE/D-final/charge_os_census.json" "$PIPE/E_assign.obs.sha256"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }; done
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench:$DS_CODE/research/campaigns/ds" JK_MAIN="$DS_CODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$DS_OUT"; cd "$DS_CODE"
N="$HOME/scripts/jacobik/notify.py"; notify() { "$JK_PY" "$N" "$@" >/dev/null 2>&1 || true; }
echo "commit $(cat COMMIT)  node $(hostname)  start $(date -Is)" | tee "$DS_OUT/job.txt"
S=research/campaigns/ds/stress_run.py; SC="observed_input_error parent_removal"
notify msg "job $SLURM_JOB_ID node $(hostname) qos regular: DS stress pilot start (LUQVOK Q1 + LIPDEU QO, refit stage), cap 30 min"
( while sleep 300; do notify msg "job $SLURM_JOB_ID DS stress running: $(tail -qn 1 "$DS_OUT"/*.log | tr '\n' ' ')"; done ) & watcher=$!
/usr/bin/time -v "$JK_PY" $S refit "$PIPE" "$DS_OUT/luqvok_q1.entries.jsonl" --case LUQVOK --quantity Q1 \
  --key "('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')" --workers 48 --scenarios $SC > "$DS_OUT/luqvok_q1.log" 2>&1 &
p1=$!
/usr/bin/time -v "$JK_PY" $S refit "$PIPE" "$DS_OUT/lipdeu_qo.entries.jsonl" --case LIPDEU --quantity QO --key "[2, 4]" --workers 48 --scenarios $SC \
  > "$DS_OUT/lipdeu_qo.log" 2>&1 &
p2=$!
rc=0; wait $p1 || { rc=1; echo "LUQVOK refit failed" | tee -a "$DS_OUT/job.txt"; }
wait $p2 || { rc=1; echo "LIPDEU refit failed" | tee -a "$DS_OUT/job.txt"; }
kill "$watcher" 2>/dev/null
echo "done rc=$rc $(date -Is)" | tee -a "$DS_OUT/job.txt"
notify file "$DS_OUT/job.txt" "$DS_OUT/luqvok_q1.log" "$DS_OUT/lipdeu_qo.log" --text "job $SLURM_JOB_ID DS stress pilot end rc=$rc"
exit "$rc"
