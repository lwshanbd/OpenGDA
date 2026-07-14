#!/bin/bash
# Build omp_pingpong with clang-21 and link the matching prebuilt GiOMP library.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-/p/lustre2/shan4/new-gicc}"
export GIOMP_CXX="${GIOMP_CXX:-/p/lustre2/shan4/softwares/diomp/bin/clang++}"
export GIOMP_ROCM_ROOT="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.3}"
export GIOMP_OMP_LIBDIR="${GIOMP_OMP_LIBDIR:-/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu}"
export GIOMP_LIBDIR="${GIOMP_LIBDIR:-${GICC_ROOT}/build_ofi/lib-clang21}"
if [[ ! -f "${GIOMP_LIBDIR}/libgicc_omp.so" ]]; then
    GICC_ROOT="${GICC_ROOT}" GIOMP_OUTDIR="${GIOMP_LIBDIR}" \
        bash "${GICC_ROOT}/omp/build_libgicc_omp.sh"
fi
exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/omp_pingpong.cpp" \
    "${GICC_ROOT}/build_ofi/omp_pingpong"
