#!/bin/bash
# tools/gicc-passes/tests/sweep/sweep_W.sh
# W sweep on barrier_bench. Run inside an existing salloc allocation.
# Sweep: W in {1,2,4,8,16,32}, ranks in {8,16,32}, 3 reps each.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
DATA_CSV="${DATA_CSV:-${GICC_ROOT}/tools/gicc-passes/tests/sweep/data.csv}"
EXTRACT="${GICC_ROOT}/tools/gicc-passes/tests/sweep/extract_latency.py"

cd "${GICC_ROOT}/build_ofi"
make -j barrier_bench >/dev/null 2>&1

# Header (only if file does not exist yet)
if [[ ! -f "${DATA_CSV}" ]]; then
    echo "workload,grid,ranks,W,streams,run_idx,latency_ms" > "${DATA_CSV}"
fi

for ranks in 8 16 32; do
    case ${ranks} in
        8)  N=1; nn=8;;
        16) N=2; nn=8;;
        32) N=4; nn=8;;
    esac
    for W in 1 2 4 8 16 32; do
        for run in 1 2 3; do
            echo "[sweep_W] ranks=${ranks} W=${W} run=${run}"
            out=$(FI_MR_CACHE_MAX_COUNT=0 GICC_WINDOW=${W} \
                  srun -p pci -N ${N} -n ${ranks} --ntasks-per-node=${nn} -t 2 \
                  ./examples/ofi/barrier_bench --gicc 1000 2>&1) || true
            row=$(echo "${out}" | "${EXTRACT}" barrier_bench na ${ranks} ${W} 1 ${run})
            echo "  ${row}"
            echo "${row}" >> "${DATA_CSV}"
        done
    done
done

echo "=== sweep_W done; CSV at ${DATA_CSV} ==="
