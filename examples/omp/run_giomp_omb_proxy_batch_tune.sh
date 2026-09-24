#!/bin/bash
# Tune the proxy worker's submit batch for small OMB messages.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_proxy_batch_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_PROXY_BATCH_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_PROXY_BATCH_INNER=1 \
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
MIN_SIZE="${GIOMP_OMB_BATCH_MIN_SIZE:-4096}"
MAX_SIZE="${GIOMP_OMB_BATCH_MAX_SIZE:-65536}"
SIZES=(--min-size="${MIN_SIZE}" --max-size="${MAX_SIZE}")

for rep in 1 2 3; do
    for batch in 1 4 8 16 32 64 128; do
        "${COMMON[@]}" --env=GICC_PROXY_SUBMIT_BATCH="${batch}" \
            "${BIN}" --transport=proxy --window=64 --proxy-lanes=1 \
            --in-kernel-windows "${SIZES[@]}" \
            >"${OUT}/batch${batch}-r${rep}.log" 2>&1
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "submit_batch,rep,size_bytes,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3; do
    for batch in 1 4 8 16 32 64 128; do
        awk -F, -v b="${batch}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print b,r,$3,$5,$6,$7,$8,$9}' \
            "${OUT}/batch${batch}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
