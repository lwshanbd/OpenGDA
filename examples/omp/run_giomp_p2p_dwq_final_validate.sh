#!/bin/bash
# Same-node-set validation of the optimized Tioga GPU-trigger configuration.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/p2p_dwq_final_validate}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_P2P_DWQ_FINAL_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_P2P_DWQ_FINAL_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -c 16 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
BENCH=(--op=put --pattern=concurrent --batch=32 --outer=21 --warmup=5)

for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        dwq_modes=(sync async)
    else
        dwq_modes=(async sync)
    fi
    for mode in "${dwq_modes[@]}"; do
        async=0
        if [[ "${mode}" == "async" ]]; then async=1; fi
        "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
            --env=GICC_DWQ_CQ_THREAD=0 \
            --env=GICC_DWQ_ASYNC_STAGE="${async}" \
            "${BIN}" --transport=dwq "${BENCH[@]}" \
            >"${OUT}/dwq-${mode}-r${rep}.log" 2>&1
    done

    "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
        --env=GICC_SKIP_DWQ_INIT=1 \
        "${BIN}" --transport=mpi "${BENCH[@]}" \
        >"${OUT}/mpi-r${rep}.log" 2>&1
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "mode,rep,size_bytes,median_us_msg,mean_us_msg,effective_GBps,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for mode in dwq-sync dwq-async mpi; do
        awk -F, -v m="${mode}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print m,r,$5,$7,$8,$9,$10}' \
            "${OUT}/${mode}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
