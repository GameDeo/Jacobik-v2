#!/bin/bash
# A4b + A4t rerun, CPU stage: GFN2 relaxations, CPU MACE, A4b tables/report, A4t full-pool relaxations and analysis.
# BILLED: one CPU node (2 x 64-core EPYC, 512 GB). Expected 10-15 min; cap 30 min (<= 0.5 node-hours).
# Needs the owner's go, per job, with the QOS named. Prerequisite: the solve stage finished on the login A100
# (python -m tools.jgen.a4b.rerun --help). $SCRATCH is mounted on compute nodes: nothing is copied.
#   sbatch ~/scripts/jacobik/jk_a4_rerun_cpu.sbatch
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-a4-rerun
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -euo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"; source "$HOME/scripts/jacobik/jk_preflight.sh"
jk_preflight                                   # xtb, MACE model hash, DB + meta, fingerprint, torch/mace imports
CODE="$JK_SCR/ext/a4-rerun"                    # snapshot of infra/a4-rerun (research tools only; src == master)
SOLVE="${A4R_SOLVE:-$JK_SCR/runs/a4r-solve}"
INPUTS="$JK_SCR/data/a4b_inputs"
OUT="$JK_SCR/runs/a4r-post-$SLURM_JOB_ID"
[ -d "$SOLVE/control/records" ] || { echo "solve stage missing: $SOLVE" >&2; exit 2; }
export PYTHONPATH="$CODE/src:$CODE:$CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$OUT/tmp"; mkdir -p "$TMPDIR"
cd "$CODE"
# 128 physical cores: one single-threaded xtb (or MACE process) per core
srun -n 1 -c 256 --cpu-bind=none "$JK_PY" -m tools.jgen.a4b.rerun post \
    --solve "$SOLVE" --inputs "$INPUTS" --out "$OUT" --workers 128 2>&1 | grep -v "UserWarning: runkit.mace on CPU" | tee "$OUT.log"
rm -rf "$TMPDIR" "$OUT/a4b/relax/tmp" "$OUT/a4t/relax/tmp"
echo "done: $OUT (packed: $OUT.tar)"
