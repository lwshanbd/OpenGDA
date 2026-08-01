#!/bin/bash
# Build the OMB-derived GiOMP PUT bandwidth benchmark.
# GIOMP_BACKEND=hip is the default; use GIOMP_BACKEND=cuda with the CUDA
# compiler settings accepted by build_giomp_example.sh on NVIDIA systems.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export GIOMP_LIBDIR="${GIOMP_LIBDIR:-${GICC_ROOT}/build_ofi/lib}"

GICC_ROOT="${GICC_ROOT}" GIOMP_OUTDIR="${GIOMP_LIBDIR}" \
    bash "${GICC_ROOT}/omp/build_libgicc_omp.sh"

exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/giomp_omb_put_bw.cpp" \
    "${GICC_ROOT}/build_ofi/giomp_omb_put_bw"
