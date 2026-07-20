#!/bin/bash
# Run the fused GiOMP and split GPU-aware MPI forms on one fixed allocation.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${GIOMP_FUSED_BIN:-${GICC_ROOT}/build_ofi/giomp_fused_kernel_bench}"
MODE="${1:-quick}"
OUT="${2:-${GICC_ROOT}/build_ofi/fused_kernel_${MODE}}"
QUEUE="${GIOMP_QUEUE:-pci}"
CORES="${GIOMP_CORES_PER_RANK:-2}"

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
    echo "usage: $0 [quick|full] [output-directory]" >&2
    exit 2
fi

if [[ "${GIOMP_FUSED_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t 20m -N2 -n2 -c "${CORES}" -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_FUSED_INNER=1 \
        --env=GIOMP_CORES_PER_RANK="${CORES}" \
        bash "${BASH_SOURCE[0]}" "${MODE}" "${OUT}"
fi

if [[ ! -x "${BIN}" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_fused_kernel_bench.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t 2m -N2 -n2 -c "${CORES}" -g1 -o mpibind=off
        --env=HSA_XNACK=1
        --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512)

if [[ "${MODE}" == "quick" ]]; then
    CASES=(
        "8 500 0"
        "8 500 500"
        "4096 500 0"
        "4096 500 500"
        "4096 500 2000"
        "65536 500 2000"
    )
    RUN_ARGS=(--iterations=31 --warmup=5 --threads=256)
else
    CASES=()
    for size in 8 64 4096 65536 1048576; do
        for work_b in 0 100 500 2000 10000; do
            CASES+=("${size} 500 ${work_b}")
        done
    done
    RUN_ARGS=(--iterations=101 --warmup=20 --threads=256)
fi

for transport in proxy dwq mpi; do
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
        read -r size work_a work_b <<<"${case_spec}"
        log="${OUT}/${transport}_s${size}_a${work_a}_b${work_b}.log"
        "${COMMON[@]}" "${EXTRA[@]}" "${BIN}" \
            --transport="${transport}" --size="${size}" \
            --work-a="${work_a}" --work-b="${work_b}" \
            "${RUN_ARGS[@]}" >"${log}" 2>&1
        grep -E '^(RESULT|one_kernel_us=)' "${log}"
    done
done

{
    echo "transport,size_bytes,work_a_iterations,work_b_iterations,threads,one_kernel_us,two_kernel_us,extra_target_us,full_us,verify"
    sed -n 's/^RESULT,//p' "${OUT}"/*.log
} >"${OUT}/results.csv"

awk -F, '
    NR == 1 { next }
    {
        key = $2 "," $3 "," $4;
        full[$1 SUBSEP key] = $9;
        keys[key] = 1;
    }
    END {
        print "size_bytes,work_a_iterations,work_b_iterations,proxy_us,dwq_us,mpi_us,mpi_over_proxy,mpi_over_dwq";
        for (key in keys) {
            p = full["proxy" SUBSEP key];
            d = full["dwq" SUBSEP key];
            m = full["mpi" SUBSEP key];
            if (p != "" && d != "" && m != "")
                printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f\n", key, p, d, m, m / p, m / d;
        }
    }
' "${OUT}/results.csv" >"${OUT}/speedups.csv"

if grep -q ',FAIL$' "${OUT}/results.csv"; then
    echo "verification failure; see ${OUT}/results.csv" >&2
    exit 1
fi
echo "FUSED_KERNEL_PASS"
echo "CSV=${OUT}/results.csv"
echo "SPEEDUPS=${OUT}/speedups.csv"
