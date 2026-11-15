#!/bin/bash
# DRAFT. Submit only after the owner's per-job go for QOS regular.
# Estimate 24–32 minutes for two rounds of two 64-worker builds.
# DS_CODE must be a clean git checkout of the committed package.
# DS_OUT must be a fresh output directory on scratch.
#SBATCH -A m5184
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -J jk-ds-data
set -euo pipefail
: "${DS_CODE:?set DS_CODE to the committed checkout}"
: "${DS_OUT:?set DS_OUT to a fresh scratch output directory}"
: "${DS_PYTHON:?set DS_PYTHON to the Perlmutter environment python}"
git -C "$DS_CODE" rev-parse --verify HEAD
test -z "$(git -C "$DS_CODE" status --porcelain -- src dev research/bench)"
test ! -e "$DS_OUT"
mkdir -p "$DS_OUT"
export DS_OUT
export PYTHONPATH="$DS_CODE/src:$DS_CODE/dev:$DS_CODE:$DS_CODE/research/bench"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export DS_DATA="${SCRATCH:?}/jacobik/data"
cd "$DS_CODE"
"$DS_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ["DS_OUT"])
for name, donor, sample in (("D0",False,False),("D-data",True,False),
                            ("D-sample",False,True),("D-both",True,True)):
    (root/f"{name}.flags.json").write_text(json.dumps(
        {"donor_identity_reconcile":donor,"stable_ancestor_sampling":sample}))
PY
build() {
    name=$1
    extra=()
    if [ "$name" = D0 ]; then extra+=(--check-reproduction); fi
    srun --exclusive -n 1 -c 128 "$DS_PYTHON" -m fit.build_db \
        --profile d2 --flags "$DS_OUT/$name.flags.json" --workers 64 \
        --ref-dir "$DS_DATA/ref" --out-dir "$DS_OUT/$name" \
        --locked-path "$DS_CODE/research/bench/benchmarks/locked_exclusions.json" \
        --fit-frame-path "$DS_DATA/refit/frame_v1/frame.json" \
        --target-db "$DS_DATA/db_d2/constraints_2026-11-09.json" "${extra[@]}" \
        > "$DS_OUT/$name.log" 2>&1
}
result=0
for pair in "D0 D-data" "D-sample D-both"; do
    pids=()
    for name in $pair; do build "$name" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid" || result=1; done
    if [ "$result" != 0 ]; then exit "$result"; fi
done
exit "$result"
