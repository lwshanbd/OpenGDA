#!/bin/bash
# Compare strict one-PUT/one-completion latency using OMB latency counts.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_put_latency}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_PUT_LAT_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_PUT_LAT_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)
SIZES=(--min-size=4096 --max-size=4194304)

for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        proxy_modes=(host-window in-kernel)
    else
        proxy_modes=(in-kernel host-window)
    fi
    for mode in "${proxy_modes[@]}"; do
        args=(--transport=proxy --latency --proxy-lanes=1 "${SIZES[@]}")
        if [[ "${mode}" == "in-kernel" ]]; then
            args+=(--in-kernel-windows)
        fi
        "${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 \
            --env=GICC_SKIP_DWQ_INIT=1 --env=GICC_HALO_DWQ=0 \
            --env=GICC_NUM_PROXY_THREADS=1 \
            "${BIN}" "${args[@]}" >"${OUT}/proxy-${mode}-r${rep}.log" 2>&1
    done

    "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
        "${BIN}" --transport=gpu-trigger --latency "${SIZES[@]}" \
        >"${OUT}/gpu-trigger-r${rep}.log" 2>&1

    "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
        --env=GICC_SKIP_DWQ_INIT=1 \
        "${BIN}" --transport=mpi-rma --latency "${SIZES[@]}" \
        >"${OUT}/mpi-rma-r${rep}.log" 2>&1
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "mode,rep,size_bytes,iterations,warmup,latency_us,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for mode in proxy-host-window proxy-in-kernel gpu-trigger mpi-rma; do
        awk -F, -v m="${mode}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print m,r,$3,$5,$6,$8,$9}' \
            "${OUT}/${mode}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
