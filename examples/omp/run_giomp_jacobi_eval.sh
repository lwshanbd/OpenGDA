#!/bin/bash
# Weak-scale GiOMP, DiOMP, and GPU-aware MPI Jacobi on one allocation.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN_DIR="${GIOMP_JACOBI_BIN_DIR:-${GICC_ROOT}/build_ofi/jacobi_eval}"
MODE="${1:-quick}"
OUT="${2:-${GICC_ROOT}/build_ofi/jacobi_${MODE}}"
QUEUE="${GIOMP_QUEUE:-pci}"
BACKENDS_SPEC="${GIOMP_JACOBI_BACKENDS:-giomp diomp mpi}"
read -r -a BACKENDS <<<"${BACKENDS_SPEC}"

case "${MODE}" in
    smoke) RANKS=(2); RUNS=1; WARMUP=0; ITERS=10; ALLOC_NODES=2 ;;
    quick) RANKS=(8 16); RUNS=3; WARMUP=1; ITERS=100; ALLOC_NODES=2 ;;
    full)  RANKS=(8 16 32); RUNS=5; WARMUP=2; ITERS=200; ALLOC_NODES=4 ;;
    paper) RANKS=(1 2 4 8 16 32 64); RUNS=5; WARMUP=2; ITERS=200; ALLOC_NODES=8 ;;
    *) echo "usage: $0 [smoke|quick|full|paper] [output-directory]" >&2; exit 2 ;;
esac

RANKS_SPEC="${GIOMP_JACOBI_RANKS:-}"
if [[ -n "${RANKS_SPEC}" ]]; then
    read -r -a RANKS <<<"${RANKS_SPEC}"
fi
ALLOC_NODES="${GIOMP_JACOBI_ALLOC_NODES:-${ALLOC_NODES}}"
ALLOC_TIME="${GIOMP_JACOBI_ALLOC_TIME:-60m}"
CASE_TIME="${GIOMP_JACOBI_CASE_TIME:-10m}"
NX="${GIOMP_JACOBI_NX:-2048}"
ROWS="${GIOMP_JACOBI_ROWS:-256}"
RUNS="${GIOMP_JACOBI_RUNS:-${RUNS}}"
WARMUP="${GIOMP_JACOBI_WARMUP:-${WARMUP}}"
ITERS="${GIOMP_JACOBI_ITERS:-${ITERS}}"
export LD_LIBRARY_PATH="/p/lustre2/shan4/softwares/diomp/lib:/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu:/opt/rocm-6.4.3/lib:/opt/cray/pe/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${GIOMP_JACOBI_INNER:-0}" != "1" ]]; then
    exec flux alloc -q "${QUEUE}" -t "${ALLOC_TIME}" -N "${ALLOC_NODES}" \
        -n "$((ALLOC_NODES * 8))" -c2 -g1 --cwd="${GICC_ROOT}" \
        --env=GIOMP_JACOBI_INNER=1 --env=GIOMP_JACOBI_BACKENDS="${BACKENDS_SPEC}" \
        --env=GIOMP_JACOBI_RANKS="${RANKS_SPEC}" \
        --env=GIOMP_JACOBI_ALLOC_NODES="${ALLOC_NODES}" \
        --env=GIOMP_JACOBI_ALLOC_TIME="${ALLOC_TIME}" \
        --env=GIOMP_JACOBI_CASE_TIME="${CASE_TIME}" \
        --env=GIOMP_JACOBI_NX="${NX}" --env=GIOMP_JACOBI_ROWS="${ROWS}" \
        --env=GIOMP_JACOBI_RUNS="${RUNS}" --env=GIOMP_JACOBI_WARMUP="${WARMUP}" \
        --env=GIOMP_JACOBI_ITERS="${ITERS}" --env=LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        bash "${BASH_SOURCE[0]}" "${MODE}" "${OUT}"
fi

if [[ ! -x "${BIN_DIR}/giomp_jacobi_eval" ||
      ! -x "${BIN_DIR}/diomp_jacobi_eval" ||
      ! -x "${BIN_DIR}/mpi_jacobi_eval" ]]; then
    bash "${GICC_ROOT}/examples/omp/build_giomp_jacobi_eval.sh"
fi
mkdir -p "${OUT}"

COMMON=(flux run -t "${CASE_TIME}" -c2 -g1 -o mpibind=off
        --env=HSA_XNACK=1 --env=FI_MR_CACHE_MAX_COUNT=0
        --env=PMI_MAX_KVS_ENTRIES=512 --env=ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
ARGS=(--nx="${NX}" --local-rows="${ROWS}" --iterations="${ITERS}"
      --runs="${RUNS}" --warmup="${WARMUP}" --teams=64 --threads=256)

for ranks in "${RANKS[@]}"; do
    nodes=$(((ranks + 7) / 8))
    for backend in "${BACKENDS[@]}"; do
        extra=()
        case "${backend}" in
            giomp)
                bin="${BIN_DIR}/giomp_jacobi_eval"
                extra=(--env=GICC_HALO_IPC=1 --env=GICC_PROXY_ENABLED=1
                       --env=GICC_SKIP_DWQ_INIT=1 --env=GICC_HALO_DWQ=0)
                ;;
            diomp)
                bin="${BIN_DIR}/diomp_jacobi_eval"
                extra=(--env=GASNET_BACKTRACE=1)
                ;;
            mpi)
                bin="${BIN_DIR}/mpi_jacobi_eval"
                extra=(--env=MPICH_GPU_SUPPORT_ENABLED=1
                       --env=MPICH_ASYNC_PROGRESS=1)
                ;;
            *) echo "unknown backend: ${backend}" >&2; exit 2 ;;
        esac
        log="${OUT}/${backend}_r${ranks}.log"
        "${COMMON[@]}" -N "${nodes}" -n "${ranks}" "${extra[@]}" \
            "${bin}" "${ARGS[@]}" >"${log}" 2>&1
        grep -E '^(RESULT|median_us=)' "${log}"
    done
done

{
    echo "backend,ranks,nx,local_rows,iterations,runs,warmup,median_us,min_us,max_us,checksum,verify"
    sed -n 's/^RESULT,//p' "${OUT}"/*.log
} >"${OUT}/results.csv"

{
    echo "ranks,giomp_us,diomp_us,mpi_us,diomp_over_giomp,mpi_over_giomp"
    for ranks in "${RANKS[@]}"; do
        awk -F, -v wanted="${ranks}" '
            NR > 1 && $2 == wanted { t[$1] = $8; c[$1] = $11 }
            END {
                if (t["giomp"] && t["diomp"] && t["mpi"])
                    printf "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", wanted,
                        t["giomp"], t["diomp"], t["mpi"],
                        t["diomp"] / t["giomp"], t["mpi"] / t["giomp"];
            }
        ' "${OUT}/results.csv"
    done
} >"${OUT}/speedups.csv"

if grep -q ',FAIL$' "${OUT}/results.csv"; then
    echo "Jacobi verification failure" >&2
    exit 1
fi
awk -F, '
    NR > 1 {
        key = $2;
        if (!(key in reference)) reference[key] = $11;
        delta = $11 - reference[key];
        if (delta < 0) delta = -delta;
        scale = reference[key] < 0 ? -reference[key] : reference[key];
        if (scale < 1.0) scale = 1.0;
        if (delta / scale > 1.0e-10) {
            printf "checksum mismatch at %s ranks: %.12e versus %.12e\n", key, $11, reference[key] > "/dev/stderr";
            bad = 1;
        }
    }
    END { exit bad }
' "${OUT}/results.csv"
echo "JACOBI_EVAL_PASS"
echo "CSV=${OUT}/results.csv"
echo "SPEEDUPS=${OUT}/speedups.csv"
