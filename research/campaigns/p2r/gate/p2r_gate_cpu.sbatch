#!/bin/bash
# P2R fold-in gate, job 2 (relaxations and analysis): ONE CPU node (2 x 64 cores, 256 threads), debug QOS, cap 30 min
# (owner go 2026-11-14). Both parts run side by side on the node (about 220 single-threaded workers in all).
# Gate (registered): per arm `a4b.rerun post --select a4c --skip-a4t` (GFN2 crude, <= 5 rows per unit, CPU MACE,
#   A4c evaluation), then `a4c.relax_level --level normal`, then `a4c.gate` (the six contract gates).
# Class analysis (reported, not gated; owner 2026-11-14): `a4c.class_basins` on a stratified parent subset at seed 31, every pool
#   row relaxed once at normal level with the crude checkpoint read from the same trajectory; budget 1.0 x the gate's
#   relaxation cost estimate (total relaxation <= 2 x the gate metric).
# Submit after job 1: GATE_JOB=<job 1 id> sbatch --export=ALL research/campaigns/p2r/gate/p2r_gate_cpu.sbatch
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-p2r-gate-cpu
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
export JK_FP=d5e2f4113497 JK_DB=/pscratch/sd/a/achen310/jacobik/data/db/constraints_2026-11-12.npz
export JK_SRC="${P2R_CODE:-/pscratch/sd/a/achen310/jacobik/ext/p2r-gate}"
source "$HOME/scripts/jacobik/jk_env.sh"; source "$HOME/scripts/jacobik/jk_preflight.sh"
jk_preflight || exit 3
RUN="$JK_SCR/runs/p2r-gate-${GATE_JOB:?set GATE_JOB to job 1}"
SOLVE="$RUN/solve"; INPUTS="$JK_SCR/data/a4b_inputs"; OUT="$RUN/post-$SLURM_JOB_ID"
[ -d "$SOLVE/C/records" ] && [ -d "$SOLVE/P/records" ] || { echo "solve stage missing: $SOLVE" >&2; exit 2; }
mkdir -p "$OUT/tmp"; cd "$JK_SRC"
export PYTHONPATH="$JK_SRC/src:$JK_SRC/dev:$JK_SRC:$JK_SRC/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$OUT/tmp"
export JACOBIK_A4_SEEDS=31,32,33              # the A4 tables default to the dev seeds 11-13
NPY="$HOME/scripts/jacobik/notify.py"; T0=$(date +%s)
NOTIFY() { [ -f "$NPY" ] && timeout 90 "$JK_PY" "$NPY" "$@" || true; }
trap 'NOTIFY file "$OUT/gate.log" --text "job $SLURM_JOB_ID ENDED after $(( ($(date +%s)-T0)/60 )) min | $(tail -n 1 $OUT/class.log 2>/dev/null)"' EXIT
NOTIFY msg "job $SLURM_JOB_ID STARTED (QOS ${SLURM_JOB_QOS:-?}), P2R gate post + class analysis on $SLURM_JOB_NODELIST"

gate_node() {
  for arm in C P; do
    fp=d5e2f4113497; [ $arm = P ] && fp=a47283387172
    "$JK_PY" -m tools.jgen.a4b.rerun post --select a4c --skip-a4t --no-pack --arm $arm --fingerprint $fp \
      --solve "$SOLVE" --inputs "$INPUTS" --out "$OUT/crude_$arm" --workers 48 > "$OUT/crude_$arm.log" 2>&1 &
  done; wait
  for arm in C P; do
    "$JK_PY" -m tools.jgen.a4c.relax_level --src "$OUT/crude_$arm/a4b" --out "$OUT/normal_$arm" --level normal \
      --workers 48 --mace-device cpu > "$OUT/normal_$arm.log" 2>&1 &
  done; wait
  "$JK_PY" -m tools.jgen.a4c.gate --crude C="$OUT/crude_C" P="$OUT/crude_P" --normal C="$OUT/normal_C" P="$OUT/normal_P" \
    --records C="$SOLVE/C/records" P="$SOLVE/P/records" \
    --eligibility research/campaigns/ds/gate_eligibility_v1.json --out "$OUT/gate.json"
}
class_node() {
  "$JK_PY" -m tools.jgen.a4c.class_basins select --solve "$SOLVE" --inputs "$INPUTS" --out "$OUT/class" --arms C P --seed 31 &&
  "$JK_PY" -m tools.jgen.a4c.class_basins relax --solve "$SOLVE" --inputs "$INPUTS" --out "$OUT/class" --workers 128 &&
  "$JK_PY" -m tools.jgen.a4c.class_basins report --out "$OUT/class" --workers 64 > "$OUT/class/report.txt"
  echo "class analysis rc $? at $(( $(date +%s)-T0 )) s"
}
gate_node > "$OUT/gate.log" 2>&1 & P1=$!
P2=""; [ "${SKIP_CLASS:-0}" = 1 ] || { class_node > "$OUT/class.log" 2>&1 & P2=$!; }   # SKIP_CLASS=1: gate only
wait $P1 $P2
rm -rf "$OUT/tmp" "$OUT"/crude_*/a4b/relax/tmp "$OUT"/normal_*/tmp "$OUT/class/tmp"
echo "done $(( $(date +%s)-T0 )) s: $OUT" | tee -a "$OUT/gate.log"
