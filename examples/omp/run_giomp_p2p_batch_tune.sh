#!/bin/bash
# Tune the concurrent-window size for GPU-trigger and MPI on one node pair.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/p2p_batch_tune}"
QUEUE="${GIOMP_QUEUE:-pdebug}"

if [[ "${GIOMP_P2P_BATCH_TUNE_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c 16 -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_P2P_BATCH_TUNE_INNER=1 \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 8m -N2 -n2 -c 16 -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=PMI_MAX_KVS_ENTRIES=512)

for rep in 1 2 3 4 5; do
    for batch in 8 16 32 64 128 256 512; do
        if (( (rep + batch) % 2 == 1 )); then
            modes=(dwq mpi)
        else
            modes=(mpi dwq)
        fi
        for mode in "${modes[@]}"; do
            if [[ "${mode}" == "dwq" ]]; then
                "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
                    --env=GICC_DWQ_CQ_THREAD=0 \
                    --env=GICC_DWQ_ASYNC_STAGE=1 \
                    "${BIN}" --transport=dwq --op=put --pattern=concurrent \
                    --batch="${batch}" --outer=21 --warmup=5 \
                    >"${OUT}/${mode}-b${batch}-r${rep}.log" 2>&1
            else
                "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
                    --env=GICC_SKIP_DWQ_INIT=1 \
                    "${BIN}" --transport=mpi --op=put --pattern=concurrent \
                    --batch="${batch}" --outer=21 --warmup=5 \
                    >"${OUT}/${mode}-b${batch}-r${rep}.log" 2>&1
            fi
        done
    done
done

for log in "${OUT}"/*.log; do
    grep -q '^ALL_CASES_PASS' "${log}"
done

echo "mode,batch_setting,rep,size_bytes,effective_batch,median_us_msg,mean_us_msg,effective_GBps,verify" \
    >"${OUT}/results.csv"
for rep in 1 2 3 4 5; do
    for batch in 8 16 32 64 128 256 512; do
        for mode in dwq mpi; do
            awk -F, -v m="${mode}" -v b="${batch}" -v r="${rep}" \
                'BEGIN{OFS=","} $1=="RESULT" {print m,b,r,$5,$6,$7,$8,$9,$10}' \
                "${OUT}/${mode}-b${batch}-r${rep}.log" >>"${OUT}/results.csv"
        done
    done
done

echo "CSV: ${OUT}/results.csv"
