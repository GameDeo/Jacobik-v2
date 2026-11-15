#!/bin/bash
# P2R Step 2 baseline, CPU stage: GFN2 crude relaxations of the A4c relax set (<= 5 rows per unit), CPU MACE, A4b tables,
# A4c evaluation (OUT/a4c/summary.json), A4t full-pool relaxations of cal parents missed on every seed, A4t analysis.
# BILLED: one CPU node (2 x 64-core EPYC), debug QOS, cap 30 min (<= 0.5 node-hours). The 2026-11-08 A4 rerun post took
# 9.5 min on this node type. Needs the owner's go, per job, with the QOS named.
# Prerequisites (login node, unbilled):
#   * code snapshot of jgen/p2r-baseline at $JK_SCR/ext/p2r-baseline (git archive; COMMIT file with the sha)
#   * solves finished: $JK_SCR/runs/p2r-solve/control/{records,xyz} loose (jk_run.sh p2r-solve panel ... --no-pack)
# Submit: sbatch research/bench/tools/jgen/a4b/pm/p2r_baseline_cpu.sbatch
# Resume (reuse an earlier job's relaxations): RESUME_JOB=<jobid> sbatch --export=ALL ...
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-p2r-base
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -euo pipefail
# production since 2026-11-12 (jk_env.sh defaults still name the 11-01 json and 6b47c3fb45a9)
export JK_FP=d5e2f4113497 JK_DB=/pscratch/sd/a/achen310/jacobik/data/db/constraints_2026-11-12.npz
# the preflight (fingerprint, MACE model) must check the code that runs, not the shared ~/repos/jacobik
export JK_SRC="${P2R_CODE:-/pscratch/sd/a/achen310/jacobik/ext/p2r-baseline}"
source "$HOME/scripts/jacobik/jk_env.sh"; source "$HOME/scripts/jacobik/jk_preflight.sh"
echo "code snapshot $(cat "$JK_SRC/COMMIT" 2>/dev/null)"
jk_preflight                                   # xtb, MACE model hash, DB + meta, fingerprint, torch/mace imports
CODE="$JK_SRC"
SOLVE="${P2R_SOLVE:-$JK_SCR/runs/p2r-solve}"
INPUTS="$JK_SCR/data/a4b_inputs"
OUT="$JK_SCR/runs/p2r-post-$SLURM_JOB_ID"
[ -d "$SOLVE/control/records" ] || { echo "solve stage missing: $SOLVE" >&2; exit 2; }
mkdir -p "$OUT/tmp"

# Discord notices (owner 2026-11-10): start, every 10 min, end with the log attached. Plain text.
NPY="$HOME/scripts/jacobik/notify.py"; T0=$(date +%s); WATCH=""
NOTIFY() { [ -f "$NPY" ] && timeout 90 "$JK_PY" "$NPY" "$@" || true; }
notify_end() {
  local rc=$?; [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
  NOTIFY file "$OUT.log" --text "job $SLURM_JOB_ID $([ $rc = 0 ] && echo DONE || echo FAILED) rc=$rc after $(( ($(date +%s)-T0)/60 )) min | $(tail -n 1 "$OUT.log" 2>/dev/null)"
}
trap notify_end EXIT
NOTIFY msg "job $SLURM_JOB_ID STARTED on $(hostname) (QOS ${SLURM_JOB_QOS:-?}), P2R baseline post stage"
( while sleep 600; do NOTIFY msg "job $SLURM_JOB_ID $(( ($(date +%s)-T0)/60 )) min | $(grep '^\[a4-rerun' "$OUT.log" 2>/dev/null | tail -n 1)"; done ) &
WATCH=$!

export PYTHONPATH="$CODE/src:$CODE:$CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$OUT/tmp"
RES=(); [ -n "${RESUME_JOB:-}" ] && RES=(--resume-from "$JK_SCR/runs/p2r-post-$RESUME_JOB")
cd "$CODE"
# 128 physical cores: one single-threaded xtb (or MACE process) per core
srun -n 1 -c 256 --cpu-bind=none "$JK_PY" -m tools.jgen.a4b.rerun post --select a4c \
    --solve "$SOLVE" --inputs "$INPUTS" --out "$OUT" --workers 128 --analysis2 --no-pack "${RES[@]}" 2>&1 \
    | grep -v "UserWarning: runkit.mace on CPU" > "$OUT.log"
rm -rf "$TMPDIR" "$OUT/a4b/relax/tmp" "$OUT/a4t/relax/tmp"
echo "done: $OUT" >> "$OUT.log"
