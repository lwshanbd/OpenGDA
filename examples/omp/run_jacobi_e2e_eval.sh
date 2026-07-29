#!/bin/bash
# Reproduce the compiler-lowered Jacobi attribution study on two AMD GPUs.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${1:-${GICC_ROOT}/build_ofi/jacobi_e2e_eval}"
QUEUE="${GIOMP_QUEUE:-pci}"
TRIALS="${GIOMP_E2E_TRIALS:-3}"
LAUNCH_N="${GIOMP_E2E_LAUNCH_N:-300}"
CORES="${GIOMP_E2E_CORES_PER_RANK:-2}"

if [[ "${GIOMP_E2E_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c "${CORES}" -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_E2E_INNER=1 \
        --env=GIOMP_E2E_TRIALS="${TRIALS}" \
        --env=GIOMP_E2E_LAUNCH_N="${LAUNCH_N}" \
        --env=GIOMP_E2E_CORES_PER_RANK="${CORES}" \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

if [[ ! -x "${GICC_ROOT}/build_ofi/jacobi_e2e_proxy" ||
      ! -x "${GICC_ROOT}/build_ofi/jacobi_e2e_dwq" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_jacobi_e2e.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t 5m -N2 -n2 -c "${CORES}" -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512
        --env=JAC_ROWS=2048
        --env=JAC_ITERS=200
        --env=JAC_WARM=20
        --env=JAC_LAUNCH_N="${LAUNCH_N}")

for transport in proxy dwq; do
    if [[ "${transport}" == "proxy" ]]; then
        BIN="${GICC_ROOT}/build_ofi/jacobi_e2e_proxy"
        EXTRA=(--env=GICC_PROXY_ENABLED=1
               --env=GICC_SKIP_DWQ_INIT=1
               --env=GICC_HALO_DWQ=0)
    else
        BIN="${GICC_ROOT}/build_ofi/jacobi_e2e_dwq"
        # This kernel has no independent work after the trigger.  Let reset()
        # be the only CQ reader; the fused-pipeline benchmark deliberately
        # retains the background poller to progress communication during B.
        EXTRA=(--env=GICC_HALO_DWQ=1
               --env=GICC_DWQ_CQ_THREAD=0
               --env=GICC_DWQ_ASYNC_STAGE=0)
    fi
    for cols in 256 4096 65536; do
        for trial in $(seq 1 "${TRIALS}"); do
            log="${OUT}/${transport}_c${cols}_t${trial}.log"
            "${COMMON[@]}" "${EXTRA[@]}" --env=JAC_COLS="${cols}" \
                "${BIN}" >"${log}" 2>&1
            grep -E '^(RESULT|LAUNCH)' "${log}"
        done
    done
done

if grep -R -q 'field_hash=.*' "${OUT}" &&
   [[ "$(sed -n 's/.*field_hash=\([0-9a-f]*\).*/\1/p' "${OUT}"/*.log | sort -u | wc -l)" -le 3 ]]; then
    echo "JACOBI_E2E_COMPLETE"
else
    echo "missing correctness hashes; see ${OUT}" >&2
    exit 1
fi
