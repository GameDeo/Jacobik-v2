#!/bin/bash
# D2 rescore only: retry a failed GFN2 optimization at elevated electronic temperature.
# The caller logs retries via stderr and must compare results with the original aggregates.
set -u
: "${D2_XTB_REAL:?Set D2_XTB_REAL to the real xtb binary}"
"$D2_XTB_REAL" "$@"
rc=$?
if [ "$rc" -ne 0 ] || [ ! -s xtbopt.xyz ]; then
    echo "D2_RESCORE_XTB_RETRY etemp=1000 initial_rc=$rc cwd=$PWD" >> "${D2_XTB_RETRY_LOG:?Set D2_XTB_RETRY_LOG}"
    rm -f xtbopt.xyz
    "$D2_XTB_REAL" "$@" --etemp 1000
    exit $?
fi
