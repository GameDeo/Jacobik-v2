#!/bin/bash
# D2: production-reproduction build + D2 build + independent D2 rebuild, one CPU node, all three concurrently.
# BILLED: one CPU node (2 x 64-core EPYC, 512 GB), debug QOS, cap 30 min (<= 0.5 node-hours). Needs the owner's go.
# Adapted from .claude/d2/jk_d2_build.sbatch to the final CLI (python -m fit.build_db). Changes against the draft:
#   * the three builds run concurrently from the start (extract is chunk-parallel, so each build can use many cores at
#     once; the draft's second stage would have added a full build time and overrun the 30 min cap);
#   * plain background processes instead of srun steps (the batch step owns the whole node); WORKERS per build (default 64);
#   * the production reproduction is reported, not gated: Q5 labels, Q6P and NCI are the owner-accepted exceptions, so
#     --check-reproduction (non-zero exit on any difference) is not passed; read d2-build-<id>/repro/build_report.json;
#   * the D2 rebuild is compared with `fit.build_db compare --exact`; the NPZ is compared with the JSON of the same build.
# Prerequisite: the snapshot $JK_SCR/ext/d2-build (COMMIT file + research/bench/benchmarks/locked_exclusions.json), staged as
# in research/archive/docs/d2/D2_PROGRESS.md, and a passing --quick run on the workstation.
#   sbatch ~/scripts/jacobik/jk_d2_build.sbatch                      (MODE=all: repro + d2_a + d2_b)
#   MODE=d2 sbatch --export=ALL ~/scripts/jacobik/jk_d2_build.sbatch (only the two D2 builds)
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J jk-d2-build
#SBATCH -o /pscratch/sd/a/achen310/jacobik/runs/slurm-%x-%j.out
#SBATCH -L scratch
set -uo pipefail
source "$HOME/scripts/jacobik/jk_env.sh"
CODE="$JK_SCR/ext/d2-build"
DATA="$JK_SCR/data"
OUT="$JK_SCR/runs/d2-build-$SLURM_JOB_ID"
MODE="${MODE:-all}"
WORKERS="${WORKERS:-64}"
[ -f "$CODE/COMMIT" ] || { echo "no snapshot at $CODE" >&2; exit 2; }
for f in "$DATA/ref/tmQM" "$DATA/ref/lnQM" "$DATA/db/constraints_2026-11-01.json" \
         "$CODE/research/bench/benchmarks/locked_exclusions.json" "$DATA/refit/frame_v1/frame.json"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
export PYTHONPATH="$CODE/src:$CODE/dev:$CODE:$CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUT"; cd "$CODE"
echo "commit $(cat COMMIT)  node $(hostname)  mode $MODE  workers/build $WORKERS  start $(date -Is)" | tee "$OUT/job.txt"

build() {  # build <name> <profile> [extra fit.build_db args]
  local name=$1 prof=$2; shift 2
  mkdir -p "$OUT/$name/tmp"
  TMPDIR="$OUT/$name/tmp" /usr/bin/time -v "$JK_PY" -m fit.build_db \
      --profile "$prof" --workers "$WORKERS" \
      --ref-dir "$DATA/ref" --out-dir "$OUT/$name" \
      --locked-path "$CODE/research/bench/benchmarks/locked_exclusions.json" \
      --fit-frame-path "$DATA/refit/frame_v1/frame.json" "$@" \
      > "$OUT/$name.log" 2>&1
  local rc=$?; rm -rf "$OUT/$name/tmp"; echo "$name exit $rc $(date -Is)" | tee -a "$OUT/job.txt"; return $rc
}
T="$DATA/db/constraints_2026-11-01.json"
pids=()
case "$MODE" in
  all)   build repro production --target-db "$T" & pids+=($!)
         build d2_a d2 --target-db "$T" & pids+=($!)
         build d2_b d2 & pids+=($!) ;;
  d2)    build d2_a d2 --target-db "$T" & pids+=($!)
         build d2_b d2 & pids+=($!) ;;
  repro) build repro production --target-db "$T" & pids+=($!) ;;
  *) echo "MODE must be all|d2|repro" >&2; exit 2 ;;
esac
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
# the D2 rebuild must match the first build entry for entry; the NPZ must be the same DB as the JSON
if [ -f "$OUT/d2_b/constraints.json" ]; then
  "$JK_PY" -m fit.build_db compare "$OUT/d2_a/constraints.json" "$OUT/d2_b/constraints.json" --exact \
      > "$OUT/d2_rebuild_compare.json" 2>&1 || { echo "D2 REBUILD DIFFERS" | tee -a "$OUT/job.txt"; rc=1; }
fi
if [ -f "$OUT/d2_a/constraints.npz" ]; then
  "$JK_PY" -m fit.build_db compare "$OUT/d2_a/constraints.npz" "$OUT/d2_a/constraints.json" --exact \
      > "$OUT/d2_npz_vs_json.json" 2>&1 || { echo "NPZ DIFFERS FROM JSON" | tee -a "$OUT/job.txt"; rc=1; }
fi
tar -cf "$OUT.tar" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "done rc=$rc: $OUT (packed $OUT.tar)  end $(date -Is)" | tee -a "$OUT/job.txt"
exit $rc
