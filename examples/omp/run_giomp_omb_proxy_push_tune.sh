#!/bin/bash
# Isolate the single-producer ring-push optimization for in-kernel OMB windows.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_proxy_push_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_PROXY_PUSH_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 15m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_PROXY_PUSH_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 6m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=GICC_PROXY_ENABLED=1
        --env=GICC_SKIP_DWQ_INIT=1
        --env=GICC_HALO_DWQ=0
        --env=GICC_NUM_PROXY_THREADS=1
        --env=PMI_MAX_KVS_ENTRIES=512)
SIZES=(--min-size=4096 --max-size=4194304)

for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        modes=(single general)
    else
        modes=(general single)
    fi
    for mode in "${modes[@]}"; do
        args=(--transport=proxy --window=64 --proxy-lanes=1
              --in-kernel-windows "${SIZES[@]}")
        if [[ "${mode}" == "general" ]]; then
            args+=(--general-producer)
        fi
        "${COMMON[@]}" "${BIN}" "${args[@]}" \
            >"${OUT}/${mode}-r${rep}.log" 2>&1
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "producer,rep,size_bytes,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for mode in single general; do
        awk -F, -v m="${mode}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print m,r,$3,$5,$6,$7,$8,$9}' \
            "${OUT}/${mode}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
