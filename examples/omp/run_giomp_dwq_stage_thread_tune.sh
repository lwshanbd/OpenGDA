#!/bin/bash
# Tune host staging concurrency for the Tioga GPU-trigger path.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/dwq_stage_thread_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_DWQ_STAGE_TUNE_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 15m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_DWQ_STAGE_TUNE_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -c 16 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=GICC_HALO_DWQ=1
        --env=PMI_MAX_KVS_ENTRIES=512)
BENCH=(--transport=dwq --op=put --pattern=concurrent
       --batch=32 --outer=21 --warmup=5)

"${COMMON[@]}" --env=GICC_DWQ_ASYNC_STAGE=0 \
    --env=GICC_DWQ_CQ_THREAD=1 "${BIN}" "${BENCH[@]}" \
    >"${OUT}/sync-cq1.log" 2>&1

"${COMMON[@]}" --env=GICC_DWQ_ASYNC_STAGE=0 \
    --env=GICC_DWQ_CQ_THREAD=0 "${BIN}" "${BENCH[@]}" \
    >"${OUT}/sync-cq0.log" 2>&1

for threads in 1 2 4 8; do
    "${COMMON[@]}" --env=GICC_DWQ_ASYNC_STAGE=1 \
        --env=GICC_DWQ_STAGE_THREADS="${threads}" \
        --env=GICC_DWQ_CQ_THREAD=0 "${BIN}" "${BENCH[@]}" \
        >"${OUT}/async${threads}-cq0.log" 2>&1
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "config,size_bytes,median_us_msg,mean_us_msg,effective_GBps,verify" \
    >"${OUT}/results.csv"
for config in sync-cq1 sync-cq0 async1-cq0 async2-cq0 async4-cq0 async8-cq0; do
    awk -F, -v c="${config}" \
        'BEGIN{OFS=","} $1=="RESULT" {print c,$5,$7,$8,$9,$10}' \
        "${OUT}/${config}.log" >>"${OUT}/results.csv"
done

echo "CSV: ${OUT}/results.csv"
