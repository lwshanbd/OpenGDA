#!/bin/bash
# Run GiOMP Proxy/DWQ, DiOMP, and GPU-aware MPI on the same allocation.
# One rank and one GPU per node deliberately exclude same-node IPC.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN_DIR="${GIOMP_MM_BIN_DIR:-${GICC_ROOT}/build_ofi/mm_eval}"
MODE="${1:-quick}"
OUT="${2:-${GICC_ROOT}/build_ofi/mm_eval_${MODE}}"
QUEUE="${GIOMP_QUEUE:-pci}"
CORES="${GIOMP_CORES_PER_RANK:-2}"
NODES="${GIOMP_MM_NODES:-2}"
RANKS="${GIOMP_MM_RANKS:-${NODES}}"
SIZE_SPEC="${GIOMP_MM_SIZES:-}"
RUNS_OVERRIDE="${GIOMP_MM_RUNS:-}"
WARMUP_OVERRIDE="${GIOMP_MM_WARMUP:-}"
BACKEND_SPEC="${GIOMP_MM_BACKENDS:-giomp-proxy giomp-dwq diomp mpi}"
read -r -a BACKENDS <<<"${BACKEND_SPEC}"
CASE_TIME="${GIOMP_MM_CASE_TIME:-10m}"
ALLOC_TIME="${GIOMP_MM_ALLOC_TIME:-45m}"
# A common 128/256/512/1024 sweep across all four backends selected 512 as the
# fastest stable global setting.  1024 helped N=512 but regressed/varied at
# N=2048, so it is intentionally not the canonical default.
THREADS="${3:-${GIOMP_MM_THREADS:-512}}"
KERNEL_STYLE="${4:-${GIOMP_MM_KERNEL_STYLE:-loop}}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
    echo "usage: $0 [smoke|quick|full] [output-directory] [threads] [loop|unified|unified-fenced|master|split]" >&2
    exit 2
fi
if [[ ! "${NODES}" =~ ^[1-9][0-9]*$ || ! "${RANKS}" =~ ^[1-9][0-9]*$ ||
      "${NODES}" -ne "${RANKS}" ]]; then
    echo "GIOMP_MM_NODES and GIOMP_MM_RANKS must be equal positive integers (one rank per node)" >&2
    exit 2
fi
if (( ${#BACKENDS[@]} == 0 )); then
    echo "GIOMP_MM_BACKENDS must name at least one backend" >&2
    exit 2
fi

export LD_LIBRARY_PATH="/p/lustre2/shan4/softwares/diomp/lib:/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu:/opt/rocm-6.4.3/lib:/opt/cray/pe/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${GIOMP_MM_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t "${ALLOC_TIME}" \
        -N "${NODES}" -n "${RANKS}" -c "${CORES}" -g1 \
        --cwd="${GICC_ROOT}" --env=GIOMP_MM_INNER=1 \
        --env=GIOMP_CORES_PER_RANK="${CORES}" \
        --env=GIOMP_MM_NODES="${NODES}" --env=GIOMP_MM_RANKS="${RANKS}" \
        --env=GIOMP_MM_SIZES="${SIZE_SPEC}" \
        --env=GIOMP_MM_RUNS="${RUNS_OVERRIDE}" \
        --env=GIOMP_MM_WARMUP="${WARMUP_OVERRIDE}" \
        --env=GIOMP_MM_BACKENDS="${BACKEND_SPEC}" \
        --env=GIOMP_MM_CASE_TIME="${CASE_TIME}" \
        --env=LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        bash "${BASH_SOURCE[0]}" "${MODE}" "${OUT}" "${THREADS}" "${KERNEL_STYLE}"
fi

if [[ ! -x "${BIN_DIR}/giomp_mm_eval" ||
      ! -x "${BIN_DIR}/diomp_mm_eval" ||
      ! -x "${BIN_DIR}/mpi_mm_eval" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_mm_eval.sh"
fi
mkdir -p "${OUT}"

case "${MODE}" in
    smoke)
        SIZES=(256)
        RUNS=1
        WARMUP=0
        ;;
    quick)
        SIZES=(512 1024 2048)
        RUNS=5
        WARMUP=2
        ;;
    full)
        SIZES=(512 1024 2048 4096)
        RUNS=11
        WARMUP=3
        ;;
esac
if [[ -n "${SIZE_SPEC}" ]]; then
    IFS=',' read -r -a SIZES <<<"${SIZE_SPEC}"
fi
for n in "${SIZES[@]}"; do
    if [[ ! "${n}" =~ ^[1-9][0-9]*$ ]]; then
        echo "every N must be a positive integer: ${n}" >&2
        exit 2
    fi
    if (( n % RANKS != 0 )); then
        echo "every N must be a positive multiple of ranks=${RANKS}: ${n}" >&2
        exit 2
    fi
done
if [[ -n "${RUNS_OVERRIDE}" ]]; then RUNS="${RUNS_OVERRIDE}"; fi
if [[ -n "${WARMUP_OVERRIDE}" ]]; then WARMUP="${WARMUP_OVERRIDE}"; fi
if [[ ! "${RUNS}" =~ ^[1-9][0-9]*$ || ! "${WARMUP}" =~ ^[0-9]+$ ]]; then
    echo "GIOMP_MM_RUNS must be positive and GIOMP_MM_WARMUP non-negative" >&2
    exit 2
fi
RUN_ARGS=(--runs="${RUNS}" --warmup="${WARMUP}" --threads="${THREADS}")

COMMON=(flux run -t "${CASE_TIME}" -N "${NODES}" -n "${RANKS}"
        -c "${CORES}" -g1 -o mpibind=off
        --env=HSA_XNACK=1 --env=GICC_HALO_IPC=0
        --env=FI_MR_CACHE_MAX_COUNT=0 --env=PMI_MAX_KVS_ENTRIES=512)

run_case() {
    local backend="$1"
    local n="$2"
    local log="${OUT}/${backend}_n${n}.log"
    local bin
    local extra=()
    local args=()
    case "${backend}" in
        giomp-proxy)
            bin="${BIN_DIR}/giomp_mm_eval"
            extra=(--env=GICC_PROXY_ENABLED=1 --env=GICC_SKIP_DWQ_INIT=1
                   --env=GICC_HALO_DWQ=0)
            args=(--transport=proxy --kernel-style="${KERNEL_STYLE}")
            ;;
        giomp-dwq)
            bin="${BIN_DIR}/giomp_mm_eval"
            extra=(--env=GICC_HALO_DWQ=1)
            args=(--transport=dwq --kernel-style="${KERNEL_STYLE}")
            ;;
        diomp)
            bin="${BIN_DIR}/diomp_mm_eval"
            extra=(--env=GASNET_BACKTRACE=1)
            ;;
        mpi)
            bin="${BIN_DIR}/mpi_mm_eval"
            extra=(--env=MPICH_GPU_SUPPORT_ENABLED=1)
            ;;
        *)
            echo "unknown backend ${backend}" >&2
            return 2
            ;;
    esac
    # mpibind=off exposes all eight GPUs after Flux processes --env rules.
    # Override inside the task so both ranks (one per node) see only local GPU
    # 0.  This also prevents DiOMP Mode 2 from misclassifying a cross-node rank
    # as a same-node HIP-IPC peer based on global_rank % visible_devices.
    "${COMMON[@]}" "${extra[@]}" -- /usr/bin/env ROCR_VISIBLE_DEVICES=0 \
        "${bin}" --n="${n}" \
        --require-one-rank-per-node "${RUN_ARGS[@]}" "${args[@]}" \
        >"${log}" 2>&1
    grep -E '^(RESULT|median_us=)' "${log}"
}

# Interleave backends at each N so temporal machine variation is less likely to
# bias one implementation's entire sweep.
for n in "${SIZES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        run_case "${backend}" "${n}"
    done
done

{
    echo "backend,N,ranks,runs,warmup,median_us,min_us,max_us,global_gflops,fingerprint,max_relative_error,verify"
    sed -n 's/^RESULT,//p' "${OUT}"/*.log
} >"${OUT}/results.csv"

{
    echo "N,giomp_proxy_us,giomp_dwq_us,diomp_us,mpi_us,diomp_over_proxy,mpi_over_proxy,diomp_over_dwq,mpi_over_dwq"
    for n in "${SIZES[@]}"; do
        awk -F, -v wanted="${n}" '
            NR > 1 && $2 == wanted { t[$1] = $6 }
            END {
                if (t["giomp-proxy"] && t["giomp-dwq"] && t["diomp"] && t["mpi"])
                    printf "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        wanted, t["giomp-proxy"], t["giomp-dwq"],
                        t["diomp"], t["mpi"],
                        t["diomp"] / t["giomp-proxy"],
                        t["mpi"] / t["giomp-proxy"],
                        t["diomp"] / t["giomp-dwq"],
                        t["mpi"] / t["giomp-dwq"]
            }
        ' "${OUT}/results.csv"
    done
} >"${OUT}/speedups.csv"

if grep -q ',FAIL$' "${OUT}/results.csv"; then
    echo "verification failure; see ${OUT}/results.csv" >&2
    exit 1
fi
echo "MM_EVAL_PASS"
echo "CSV=${OUT}/results.csv"
echo "SPEEDUPS=${OUT}/speedups.csv"
