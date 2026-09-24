#!/bin/bash
# Sweep the OMB-supported window size for MPI RMA and both GiOMP paths.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_window_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_WINDOW_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 25m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_WINDOW_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
SIZES=(--min-size=4096 --max-size=4194304)

for window in 8 16 32 64 128; do
    for threads in 1 2; do
        "${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 \
            --env=GICC_SKIP_DWQ_INIT=1 --env=GICC_HALO_DWQ=0 \
            --env=GICC_NUM_PROXY_THREADS="${threads}" \
            "${BIN}" --transport=proxy --window="${window}" \
            --proxy-lanes="${threads}" "${SIZES[@]}" \
            >"${OUT}/proxy-w${window}-t${threads}.log" 2>&1
    done

    "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
        "${BIN}" --transport=gpu-trigger --window="${window}" \
        "${SIZES[@]}" >"${OUT}/gpu-trigger-w${window}.log" 2>&1

    "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
        --env=GICC_SKIP_DWQ_INIT=1 \
        "${BIN}" --transport=mpi-rma --window="${window}" \
        "${SIZES[@]}" >"${OUT}/mpi-rma-w${window}.log" 2>&1
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "transport,window,threads,size_bytes,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for window in 8 16 32 64 128; do
    for threads in 1 2; do
        awk -F, -v w="${window}" -v t="${threads}" \
            'BEGIN{OFS=","} $1=="RESULT" {print "proxy",w,t,$3,$5,$6,$7,$8,$9}' \
            "${OUT}/proxy-w${window}-t${threads}.log" >>"${OUT}/results.csv"
    done
    awk -F, -v w="${window}" \
        'BEGIN{OFS=","} $1=="RESULT" {print "gpu-trigger",w,1,$3,$5,$6,$7,$8,$9}' \
        "${OUT}/gpu-trigger-w${window}.log" >>"${OUT}/results.csv"
    awk -F, -v w="${window}" \
        'BEGIN{OFS=","} $1=="RESULT" {print "mpi-rma",w,1,$3,$5,$6,$7,$8,$9}' \
        "${OUT}/mpi-rma-w${window}.log" >>"${OUT}/results.csv"
done

echo "CSV: ${OUT}/results.csv"
