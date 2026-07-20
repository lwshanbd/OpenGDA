#!/bin/bash
# Fixed-node small-message profiling for giomp_p2p_eval.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_P2P_BIN:-${GICC_ROOT}/build_ofi/giomp_p2p_eval}"
OUT="${1:-${GICC_ROOT}/build_ofi/p2p_profile}"
QUEUE="${GIOMP_QUEUE:-pci}"
CORES="${GIOMP_CORES_PER_RANK:-1}"

if [[ "${GIOMP_P2P_PROFILE_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c "${CORES}" -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_P2P_PROFILE_INNER=1 \
        --env=GIOMP_CORES_PER_RANK="${CORES}" \
        bash "${BASH_SOURCE[0]}" "${OUT}"
fi

mkdir -p "${OUT}"
COMMON=(flux run -t 10m -N2 -n2 -c "${CORES}" -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512)
BENCH=(--op=put --pattern=concurrent --size=8 --outer=101 --warmup=20)
BATCHES=(1 2 4 8 16 32 64)
if [[ "${GIOMP_PROFILE_SHORT:-0}" == "1" ]]; then BATCHES=(1 32); fi

"${COMMON[@]}" bash -c \
    'echo "host=$(hostname) pid=$$"; grep Cpus_allowed_list /proc/self/status' \
    >"${OUT}/affinity.log" 2>&1

for batch in "${BATCHES[@]}"; do
    "${COMMON[@]}" --env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1 \
        --env=GICC_HALO_DWQ=0 --env=GICC_PROXY_PROFILE=1 \
        "${BIN}" --transport=proxy "${BENCH[@]}" --batch="${batch}" \
        --profile-phases >"${OUT}/proxy_b${batch}.log" 2>&1

    "${COMMON[@]}" --env=GICC_HALO_DWQ=1 \
        "${BIN}" --transport=dwq "${BENCH[@]}" --batch="${batch}" \
        --profile-phases >"${OUT}/dwq_b${batch}.log" 2>&1

    "${COMMON[@]}" --env=MPICH_GPU_SUPPORT_ENABLED=1 \
        "${BIN}" --transport=mpi "${BENCH[@]}" --batch="${batch}" \
        >"${OUT}/mpi_b${batch}.log" 2>&1
done

for transport in proxy dwq mpi; do
    extra=()
    if [[ "${transport}" == "proxy" ]]; then
        extra=(--env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1
               --env=GICC_HALO_DWQ=0 --env=GICC_PROXY_PROFILE=1)
    elif [[ "${transport}" == "dwq" ]]; then
        extra=(--env=GICC_HALO_DWQ=1)
    else
        extra=(--env=MPICH_GPU_SUPPORT_ENABLED=1)
    fi
    profile_arg=()
    if [[ "${transport}" != "mpi" ]]; then profile_arg=(--profile-phases); fi
    "${COMMON[@]}" "${extra[@]}" "${BIN}" --transport="${transport}" \
        --op=put --pattern=single --size=8 --batch=32 --outer=101 --warmup=20 \
        "${profile_arg[@]}" >"${OUT}/${transport}_single.log" 2>&1
done

{
    echo "transport,batch,median_us_msg,mean_us_msg"
    for transport in proxy dwq mpi; do
        for batch in "${BATCHES[@]}"; do
            sed -n "s/^RESULT,[^,]*,put,concurrent,8,${batch},\([^,]*\),\([^,]*\),.*/${transport},${batch},\1,\2/p" \
                "${OUT}/${transport}_b${batch}.log"
        done
    done
} >"${OUT}/batch_sweep.csv"

grep -H -E '^(CONTROL_PROFILE|PROFILE|RESULT)|^\[proxy-profile\]|submit->completion|max_inflight|CQ-poll' \
    "${OUT}"/*.log >"${OUT}/profile_summary.txt"
echo "PROFILE_OUTPUT=${OUT}"
