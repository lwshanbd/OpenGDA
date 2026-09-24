#!/bin/bash
# Compare synchronous and asynchronous DWQ staging under OMB PUT semantics.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_dwq_async_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_DWQ_ASYNC_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 30m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_DWQ_ASYNC_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
SIZES=(--min-size=1048576 --max-size=4194304)

# Alternate the order each repetition so warm caches and transient system
# load do not systematically favor one staging policy.
for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        modes=(0 1)
    else
        modes=(1 0)
    fi
    for window in 8 16 64; do
        for async in "${modes[@]}"; do
            "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
                --env=GICC_DWQ_ASYNC_STAGE="${async}" \
                "${BIN}" --transport=gpu-trigger --window="${window}" \
                "${SIZES[@]}" \
                >"${OUT}/gpu-trigger-a${async}-r${rep}-w${window}.log" 2>&1
        done

        "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
            --env=GICC_SKIP_DWQ_INIT=1 \
            "${BIN}" --transport=mpi-rma --window="${window}" \
            "${SIZES[@]}" \
            >"${OUT}/mpi-rma-r${rep}-w${window}.log" 2>&1
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "transport,async,window,rep,size_bytes,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for window in 8 16 64; do
        for async in 0 1; do
            awk -F, -v a="${async}" -v w="${window}" -v r="${rep}" \
                'BEGIN{OFS=","} $1=="RESULT" {print "gpu-trigger",a,w,r,$3,$5,$6,$7,$8,$9}' \
                "${OUT}/gpu-trigger-a${async}-r${rep}-w${window}.log" \
                >>"${OUT}/results.csv"
        done
        awk -F, -v w="${window}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print "mpi-rma","",w,r,$3,$5,$6,$7,$8,$9}' \
            "${OUT}/mpi-rma-r${rep}-w${window}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
