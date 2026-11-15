#!/bin/bash
#SBATCH -A m5184_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --gpus=4
#SBATCH -t 00:10:00
#SBATCH -J jk-mps-test
#SBATCH -L scratch
# Owner approval required before submission. Run from a committed infra-fast snapshot.
set -euo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
cd "$JK_SRC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
command -v xtb >/dev/null
[ "$(nvidia-smi -L | wc -l)" -ge 4 ]
[ "$("$JK_PY" -c 'from jacobik.core import profile; print(profile.snapshot()["fingerprint"])')" = 6b47c3fb45a9 ]
OUT="$JK_SCR/runs/mps_four_gpu_${SLURM_JOB_ID}"
mkdir -p "$OUT"
if [ -f "$JK_SRC/COMMIT" ]; then
  cp "$JK_SRC/COMMIT" "$OUT/commit.txt"
else
  git -C "$JK_SRC" rev-parse HEAD > "$OUT/commit.txt"
fi
# Telemetry in each runner records power and per-lane events.
codes=(AJAQAD,AWONOR AYOGUR,AZUSIA BABNAV,BIBFUN BOCRAN,BOHWUT)
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$g" TOOLS_RUN_GPU_INDEX="$g" nice -n 5 "$JK_PY" -m tools.run panel \
    --out "$OUT/g$g" --codes "${codes[$g]}" --arms C --seeds 11 --device cuda \
    --max-lanes 2 --cpu-cap 6 --max-rss-gb 6 --mps --pack --yes > "$OUT/g$g.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
"$JK_PY" dev/tools/check/verify_mps_node.py "$OUT" || rc=1
exit "$rc"
