#!/bin/bash
# Build the Jacobi solver (NVIDIA multi-gpu-programming-models "mpi") as
# OpenMP target + GPU-aware MPI and as GiOMP, with the same kernels.
#
#   mpi           jacobi_mpi_omp.cpp, MPI_Sendrecv of the halo rows
#   giomp         jacobi_giomp.cpp, ompx_put after the sweep + ompx_fence
#   giomp_signal  as giomp, -DJC_SIGNAL: ompx_put_signal per row and a wait
#                 on the two neighbours instead of the global fence
#   giomp_pipe    as giomp, built with the pass plugin (the sweep sends the
#                 rows itself as it writes them)
#
# Usage: build_jacobi.sh [OUTDIR] [VARIANT...]   (default: build_ofi/jacobi, all)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GICC_ROOT="${GICC_ROOT:-$(cd "${HERE}/../../.." && pwd)}"
OUT="${1:-${GICC_ROOT}/build_ofi/jacobi}"
shift || true
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(mpi giomp giomp_signal giomp_pipe)
mkdir -p "${OUT}"
for v in "${VARIANTS[@]}"; do
    src=jacobi_giomp.cpp
    script=build_giomp_example.sh
    flags="-foffload-lto"
    case "${v}" in
        mpi)          src=jacobi_mpi_omp.cpp ;;
        giomp)        ;;
        giomp_signal) flags="${flags} -DJC_SIGNAL" ;;
        giomp_pipe)   script=build_giomp_after.sh; flags="" ;;
        *) echo "unknown variant ${v}" >&2; exit 2 ;;
    esac
    GICC_ROOT="${GICC_ROOT}" GIOMP_EXTRA_FLAGS="${flags} ${JC_FLAGS:-}" \
        bash "${HERE}/../${script}" "${HERE}/${src}" "${OUT}/jacobi_${v}"
done
