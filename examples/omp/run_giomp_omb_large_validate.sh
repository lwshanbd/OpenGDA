#!/bin/bash
# Repeat the promising OMB window configurations on the same Tioga nodes.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_large_validate}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_VALIDATE_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_VALIDATE_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
SIZES=(--min-size=1048576 --max-size=4194304)

for rep in 1 2 3 4 5; do
    for window in 8 16; do
        "${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 \
            --env=GICC_SKIP_DWQ_INIT=1 --env=GICC_HALO_DWQ=0 \
            --env=GICC_NUM_PROXY_THREADS=2 \
            "${BIN}" --transport=proxy --window="${window}" \
            --proxy-lanes=2 "${SIZES[@]}" \
            >"${OUT}/proxy-r${rep}-w${window}.log" 2>&1

        "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
            "${BIN}" --transport=gpu-trigger --window="${window}" \
            "${SIZES[@]}" >"${OUT}/gpu-trigger-r${rep}-w${window}.log" 2>&1

        "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
            --env=GICC_SKIP_DWQ_INIT=1 \
            "${BIN}" --transport=mpi-rma --window="${window}" \
            "${SIZES[@]}" >"${OUT}/mpi-rma-r${rep}-w${window}.log" 2>&1
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "transport,window,rep,size_bytes,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for window in 8 16; do
        for transport in proxy gpu-trigger mpi-rma; do
            awk -F, -v tr="${transport}" -v w="${window}" -v r="${rep}" \
                'BEGIN{OFS=","} $1=="RESULT" {print tr,w,r,$3,$5,$6,$7,$8,$9}' \
                "${OUT}/${transport}-r${rep}-w${window}.log" >>"${OUT}/results.csv"
        done
    done
done

echo "CSV: ${OUT}/results.csv"
