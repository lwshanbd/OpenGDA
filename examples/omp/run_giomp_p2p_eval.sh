#!/bin/bash
# Reproduce the cross-node GiOMP proxy/DWQ vs GPU-aware MPI evaluation.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/p2p_results}"
QUEUE="${GIOMP_QUEUE:-pci}"

# Allocate the two nodes once so proxy, DWQ, and MPI run on the same hardware.
# The inner invocation uses the child Flux instance created by flux alloc.
if [[ "${GIOMP_P2P_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 15m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_P2P_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

if [[ ! -x "${BIN}" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_p2p_eval.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t 10m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512)

"${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1 \
    --env=GICC_HALO_DWQ=0 "${BIN}" --transport=proxy \
    >"${OUT}/proxy.log" 2>&1

"${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
    "${BIN}" --transport=dwq >"${OUT}/dwq.log" 2>&1

"${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    "${BIN}" --transport=mpi >"${OUT}/mpi.log" 2>&1

{
    echo "transport,op,pattern,size_bytes,batch,median_us_msg,mean_us_msg,effective_GBps,verify"
    sed -n 's/^RESULT,//p' "${OUT}/proxy.log"
    sed -n 's/^RESULT,//p' "${OUT}/dwq.log"
    sed -n 's/^RESULT,//p' "${OUT}/mpi.log"
} >"${OUT}/results.csv"

grep -H '^ALL_CASES_PASS' "${OUT}"/*.log
echo "CSV: ${OUT}/results.csv"
