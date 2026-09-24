#!/bin/bash
# Isolate MR-cache, CQ-progress, and async-staging effects on one node pair.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/p2p_dwq_config_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_P2P_DWQ_CONFIG_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_P2P_DWQ_CONFIG_INNER=1 \
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

for rep in 1 2 3 4 5; do
    modes=(legacy mr-cache caller-progress optimized mr0-caller-progress)
    if (( rep % 2 == 0 )); then
        modes=(mr0-caller-progress optimized caller-progress mr-cache legacy)
    fi
    for mode in "${modes[@]}"; do
        case "${mode}" in
            legacy)
                extra=(--env=FI_MR_CACHE_MAX_COUNT=0
                       --env=GICC_DWQ_CQ_THREAD=1
                       --env=GICC_DWQ_ASYNC_STAGE=0)
                ;;
            mr-cache)
                extra=(--env=GICC_DWQ_CQ_THREAD=1
                       --env=GICC_DWQ_ASYNC_STAGE=0)
                ;;
            caller-progress)
                extra=(--env=GICC_DWQ_CQ_THREAD=0
                       --env=GICC_DWQ_ASYNC_STAGE=0)
                ;;
            optimized)
                extra=(--env=GICC_DWQ_CQ_THREAD=0
                       --env=GICC_DWQ_ASYNC_STAGE=1)
                ;;
            mr0-caller-progress)
                extra=(--env=FI_MR_CACHE_MAX_COUNT=0
                       --env=GICC_DWQ_CQ_THREAD=0
                       --env=GICC_DWQ_ASYNC_STAGE=0)
                ;;
        esac
        "${COMMON[@]}" "${extra[@]}" "${BIN}" "${BENCH[@]}" \
            >"${OUT}/${mode}-r${rep}.log" 2>&1
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "mode,rep,size_bytes,median_us_msg,mean_us_msg,effective_GBps,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for mode in legacy mr-cache caller-progress optimized mr0-caller-progress; do
        awk -F, -v m="${mode}" -v r="${rep}" \
            'BEGIN{OFS=","} $1=="RESULT" {print m,r,$5,$7,$8,$9,$10}' \
            "${OUT}/${mode}-r${rep}.log" >>"${OUT}/results.csv"
    done
done

echo "CSV: ${OUT}/results.csv"
