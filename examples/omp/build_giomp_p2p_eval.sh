#!/bin/bash
# Build the paper-oriented GiOMP vs GPU-aware MPI P2P benchmark.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export GIOMP_CXX="${GIOMP_CXX:-/opt/rocm-6.4.0/lib/llvm/bin/clang++}"
export GIOMP_ROCM_ROOT="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.0}"
export GIOMP_OMP_LIBDIR="${GIOMP_OMP_LIBDIR:-/opt/rocm-6.4.0/lib/llvm/lib}"
export GIOMP_LIBDIR="${GIOMP_LIBDIR:-${GICC_ROOT}/build_ofi/lib}"

GICC_ROOT="${GICC_ROOT}" GIOMP_OUTDIR="${GIOMP_LIBDIR}" \
    bash "${GICC_ROOT}/omp/build_libgicc_omp.sh"

exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/giomp_p2p_eval.cpp" \
    "${GICC_ROOT}/build_ofi/giomp_p2p_eval"
