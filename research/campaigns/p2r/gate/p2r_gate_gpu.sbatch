#!/bin/bash
# P2R fold-in gate, job 1 (solves): arms C (production d5e2f4113497) and P (P2R flags on, a47283387172), cal + tail,
# fresh seeds 31-33, per-parent cap 5 min. BILLED: one GPU node (4 x A100), debug QOS, cap 30 min (owner go 2026-11-14).
# Contract: research/campaigns/p2r/FOLDIN_GATE_CONTRACT.md. GPUs 0-1 run arm C, GPUs 2-3 arm P, the same two
# parent shards (recorded cost, greedy) on each pair, so per-arm parent walls are separable (gate 5) and both arms run
# on the same node. Output: $OUT/solve/{C,P}/{records,xyz} (loose files, read by job 2).
# Prerequisite: code snapshot of jgen/p2r-gate at $JK_SCR/ext/p2r-gate (git archive + COMMIT file).
#SBATCH -A m5184_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 128
#SBATCH --gpus=4
#SBATCH -t 00:30:00
#SBATCH -J jk-p2r-gate-gpu
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
export JK_FP=d5e2f4113497 JK_DB=/pscratch/sd/a/achen310/jacobik/data/db/constraints_2026-11-12.npz
export JK_SRC="${P2R_CODE:-/pscratch/sd/a/achen310/jacobik/ext/p2r-gate}"
source "$HOME/scripts/jacobik/jk_env.sh"; source "$HOME/scripts/jacobik/jk_preflight.sh"
echo "code snapshot $(cat "$JK_SRC/COMMIT" 2>/dev/null)"
jk_preflight || exit 3                         # xtb, MACE model hash, DB + meta, fingerprint, torch/mace imports
[ "$(nvidia-smi -L | wc -l)" -ge 4 ] || { echo "preflight: fewer than 4 GPUs"; exit 3; }
OUT="$JK_SCR/runs/p2r-gate-$SLURM_JOB_ID"; mkdir -p "$OUT"
cd "$JK_SRC"
export PYTHONPATH="$JK_SRC/src:$JK_SRC/dev:$JK_SRC:$JK_SRC/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi > "$OUT/nvidia-smi.txt"; cp "$JK_SRC/COMMIT" "$OUT/" 2>/dev/null

NPY="$HOME/scripts/jacobik/notify.py"; T0=$(date +%s); WATCH=""
NOTIFY() { [ -f "$NPY" ] && timeout 90 "$JK_PY" "$NPY" "$@" || true; }
trap '[ -n "$WATCH" ] && kill $WATCH 2>/dev/null; NOTIFY file "$OUT/summary.txt" --text "job $SLURM_JOB_ID ENDED after $(( ($(date +%s)-T0)/60 )) min"' EXIT
NOTIFY msg "job $SLURM_JOB_ID STARTED on $(hostname) (QOS ${SLURM_JOB_QOS:-?}), P2R gate solves"
( while sleep 600; do NOTIFY msg "job $SLURM_JOB_ID $(( ($(date +%s)-T0)/60 )) min | records C $(ls $OUT/C_g*/C/records 2>/dev/null | wc -l) P $(ls $OUT/P_g*/P/records 2>/dev/null | wc -l)"; done ) &
WATCH=$!

# parent shards: cal + tail minus locked codes, two bins by recorded cost (longest first)
"$JK_PY" - "$JK_SCR/data/a4b_inputs" > "$OUT/shards.txt" <<'PY'
import json, sys
from pathlib import Path
from tools.jgen.a4b.rerun import panel_codes
from tools.check.pm_full import shard
codes = panel_codes(Path(sys.argv[1]))
per = json.loads(Path("dev/tools/run/costs.json").read_text())["bench_c_per_seed_s"]
med = sorted(per.values())[len(per) // 2]
for b in shard({c: per.get(c, med) for c in codes}, 2):
    print(",".join(b))
PY
mapfile -t SH < "$OUT/shards.txt"
[ "${#SH[@]}" -eq 2 ] || { echo "sharding failed"; exit 4; }
SPEC_C='[{"arm":"C","flags":{}}]'
SPEC_P='[{"arm":"P","flags":{"out_mirror":true,"ring_states":true,"ring_states_five":true,"config_states":true,"config_near":true}}]'
g=0; PIDS=()
for arm in C P; do
  spec=$SPEC_C; [ $arm = P ] && spec=$SPEC_P
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES=$g TOOLS_RUN_GPU_INDEX=$g timeout 1560 nice -n 5 "$JK_PY" -m tools.run panel \
      --out "$OUT/${arm}_g$s" --name p2r_gate_${arm}_g$s --codes "${SH[$s]}" --arms-spec "$spec" --seeds 31,32,33 \
      --env JACOBIK_DB="$JK_DB" --max-lanes 4 --cpu-cap 30 --max-rss-gb 55 --mps --no-pack --yes > "$OUT/${arm}_g$s.log" 2>&1 &
    PIDS+=($!); g=$((g+1))
  done
done
wait "${PIDS[@]}"            # the runners only: a bare wait would also wait for the notice loop (job 61284799 hung)
# one solve dir per arm for the post stage
for arm in C P; do
  mkdir -p "$OUT/solve/$arm/records" "$OUT/solve/$arm/xyz"
  for s in 0 1; do
    d="$OUT/${arm}_g$s/$arm"
    [ -d "$d/records" ] && ln -sf "$d"/records/*.json "$OUT/solve/$arm/records/"
    [ -d "$d/xyz" ] && cp -al "$d"/xyz/* "$OUT/solve/$arm/xyz/"   # real dirs of hard links: Path.rglob does not follow symlinked dirs
  done
done
{ echo "job $SLURM_JOB_ID wall $(( $(date +%s)-T0 )) s"
  for arm in C P; do echo "$arm records $(ls "$OUT/solve/$arm/records" | wc -l) ok $(grep -l '"status": "ok"' "$OUT"/solve/$arm/records/*.json 2>/dev/null | wc -l)"; done
  grep -h "GFN2 embed refinement failed" "$OUT"/*_g*.log | head -3; } | tee "$OUT/summary.txt"
