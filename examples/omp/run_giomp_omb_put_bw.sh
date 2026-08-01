#!/bin/bash
# Run the OMB-derived PUT bandwidth comparison on the same two nodes.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_put_bw_results}"
QUEUE="${GIOMP_QUEUE:-pci}"
EXTRA_ARGS=()
if [[ "${GIOMP_OMB_QUICK:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quick)
fi

if [[ "${GIOMP_OMB_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

if [[ ! -x "${BIN}" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_omb_put_bw.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t 12m -N2 -n2 -c 16 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)

"${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1 \
    --env=GICC_HALO_DWQ=0 "${BIN}" --transport=proxy \
    --in-kernel-windows "${EXTRA_ARGS[@]}" \
    >"${OUT}/proxy.log" 2>&1

"${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
    --env=GICC_DWQ_CQ_THREAD=0 --env=GICC_DWQ_ASYNC_STAGE=1 \
    "${BIN}" --transport=gpu-trigger "${EXTRA_ARGS[@]}" \
    >"${OUT}/gpu-trigger.log" 2>&1

"${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=GICC_SKIP_DWQ_INIT=1 "${BIN}" --transport=mpi-rma \
    "${EXTRA_ARGS[@]}" \
    >"${OUT}/mpi-rma.log" 2>&1

{
    echo "transport,size_bytes,window,iterations,warmup,MBps,us_per_message,verify"
    sed -n 's/^RESULT,//p' "${OUT}/proxy.log"
    sed -n 's/^RESULT,//p' "${OUT}/gpu-trigger.log"
    sed -n 's/^RESULT,//p' "${OUT}/mpi-rma.log"
} >"${OUT}/results.csv"

grep -H '^ALL_CASES_PASS' "${OUT}"/*.log
echo "CSV: ${OUT}/results.csv"
