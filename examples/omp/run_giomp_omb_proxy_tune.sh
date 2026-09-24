#!/bin/bash
# Explore proxy fan-out without changing the OMB PUT-window semantics.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_OMB_PUT_BW_BIN:-${GICC_ROOT}/build_ofi/giomp_omb_put_bw}"
OUT="${1:-${GICC_ROOT}/build_ofi/omb_proxy_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_OMB_TUNE_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_OMB_TUNE_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=GICC_PROXY_ENABLED=1
        --env=GICC_SKIP_DWQ_INIT=1
        --env=GICC_HALO_DWQ=0
        --env=PMI_MAX_KVS_ENTRIES=512)

for threads in 1 2 4 8; do
    "${COMMON[@]}" --env=GICC_NUM_PROXY_THREADS="${threads}" \
        "${BIN}" --transport=proxy --window=64 \
        --min-size=4096 --max-size=4194304 \
        --proxy-lanes="${threads}" \
        >"${OUT}/sequential-t${threads}.log" 2>&1

    "${COMMON[@]}" --env=GICC_NUM_PROXY_THREADS="${threads}" \
        "${BIN}" --transport=proxy --window=64 \
        --min-size=4096 --max-size=4194304 \
        --proxy-lanes="${threads}" --parallel-proxy \
        >"${OUT}/parallel-t${threads}.log" 2>&1
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "mode,threads,transport,size_bytes,window,iterations,warmup,MBps,us_per_message,verify" \
    >"${OUT}/results.csv"
for mode in sequential parallel; do
    for threads in 1 2 4 8; do
        sed -n "s/^RESULT,/${mode},${threads},/p" \
            "${OUT}/${mode}-t${threads}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
