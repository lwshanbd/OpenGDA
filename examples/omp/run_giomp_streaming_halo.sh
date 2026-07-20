#!/bin/bash
# Run Proxy, DWQ, and optimized GPU-aware MPI on one fixed two-node allocation.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_STREAMING_BIN:-${GICC_ROOT}/build_ofi/giomp_streaming_halo}"
MODE="${1:-quick}"
OUT="${2:-${GICC_ROOT}/build_ofi/streaming_halo_${MODE}}"
QUEUE="${GIOMP_QUEUE:-pci}"
CORES="${GIOMP_CORES_PER_RANK:-2}"
TRANSPORT_SPEC="${GIOMP_TRANSPORTS:-proxy dwq mpi}"
read -r -a TRANSPORTS <<<"${TRANSPORT_SPEC}"

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
    echo "usage: $0 [quick|full] [output-directory]" >&2
    exit 2
fi

if [[ "${GIOMP_STREAMING_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c "${CORES}" -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_STREAMING_INNER=1 \
        --env=GIOMP_CORES_PER_RANK="${CORES}" \
        --env=GIOMP_TRANSPORTS="${TRANSPORT_SPEC}" \
        bash "${BASH_SOURCE[0]}" "${MODE}" "${OUT}"
fi

if [[ ! -x "${BIN}" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_streaming_halo.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t 2m -N2 -n2 -c "${CORES}" -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512)

if [[ "${MODE}" == "quick" ]]; then
    CASES=(
        "4096 1 0 0"
        "4096 1 0 10000"
        "65536 16 4000 0"
        "65536 16 4000 10000"
        "1048576 16 4000 0"
        "1048576 16 4000 10000"
    )
    RUN_ARGS=(--iterations=11 --warmup=3 --teams=32 --threads=256)
else
    CASES=()
    for size in 256 4096 65536 1048576; do
        for work in 0 500 2000 10000 40000; do
            CASES+=("${size} 1 0 ${work}")
            CASES+=("${size} 16 4000 ${work}")
        done
    done
    RUN_ARGS=(--iterations=31 --warmup=5 --teams=32 --threads=256)
fi

for transport in "${TRANSPORTS[@]}"; do
    EXTRA=()
    if [[ "${transport}" == "proxy" ]]; then
        EXTRA=(--env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1
               --env=GICC_HALO_DWQ=0)
    elif [[ "${transport}" == "dwq" ]]; then
        EXTRA=(--env=GICC_HALO_DWQ=1)
    else
        EXTRA=(--env=MPICH_GPU_SUPPORT_ENABLED=1
               --env=MPICH_ASYNC_PROGRESS=1)
    fi
    for case_spec in "${CASES[@]}"; do
        read -r size tiles producer_work work <<<"${case_spec}"
        log="${OUT}/${transport}_s${size}_t${tiles}_p${producer_work}_w${work}.log"
        "${COMMON[@]}" "${EXTRA[@]}" "${BIN}" \
            --transport="${transport}" --size="${size}" --tiles="${tiles}" \
            --producer-work="${producer_work}" --work="${work}" \
            "${RUN_ARGS[@]}" >"${log}" 2>&1
        grep -E '^(RESULT|producer_us=)' "${log}"
    done
done

{
    echo "transport,size_bytes,tiles,producer_work_iterations,interior_work_iterations,teams,threads,producer_us,compute_only_us,comm_only_us,full_us,independent_compute_us,communication_us,exposed_us,overlap_efficiency,verify"
    sed -n 's/^RESULT,//p' "${OUT}"/*.log
} >"${OUT}/results.csv"

awk -F, '
    NR == 1 { next }
    {
        key = $2 "," $3 "," $4 "," $5;
        value[$1 SUBSEP key] = $11;
        keys[key] = 1;
    }
    END {
        print "size_bytes,tiles,producer_work_iterations,interior_work_iterations,proxy_us,dwq_us,mpi_us,mpi_over_proxy,mpi_over_dwq";
        for (key in keys) {
            p = value["proxy" SUBSEP key];
            d = value["dwq" SUBSEP key];
            m = value["mpi" SUBSEP key];
            if (p != "" && d != "" && m != "")
                printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f\n", key, p, d, m, m / p, m / d;
        }
    }
' "${OUT}/results.csv" >"${OUT}/speedups.csv"

if grep -q ',FAIL$' "${OUT}/results.csv"; then
    echo "verification failure; see ${OUT}/results.csv" >&2
    exit 1
fi
echo "STREAMING_HALO_PASS"
echo "CSV=${OUT}/results.csv"
echo "SPEEDUPS=${OUT}/speedups.csv"
