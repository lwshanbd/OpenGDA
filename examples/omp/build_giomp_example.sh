#!/bin/bash
# Build one ordinary OpenMP-target example against the prebuilt GiOMP library.
# Usage: build_giomp_example.sh SOURCE OUTPUT
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SOURCE OUTPUT" >&2
    exit 2
fi

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROCM="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.0}"
CLANG="${GIOMP_CXX:-${ROCM}/lib/llvm/bin/clang++}"
LIBFAB="${GIOMP_LIBFABRIC_ROOT:-/opt/cray/libfabric/2.1}"
MPI="${GIOMP_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0}"
ARCH="${GIOMP_OFFLOAD_ARCH:-gfx90a}"
OMPLIB="${GIOMP_OMP_LIBDIR:-${ROCM}/lib/llvm/lib}"
CRAYPE="${GIOMP_CRAYPE_LIBDIR:-/opt/cray/pe/lib64}"
LIBDIR="${GIOMP_LIBDIR:-${GICC_ROOT}/build_ofi/lib}"
SOURCE="$1"
OUTPUT="$2"

if [[ ! -f "${LIBDIR}/libgicc_omp.so" ]]; then
    echo "error: ${LIBDIR}/libgicc_omp.so is missing" >&2
    echo "build it first with: GICC_ROOT=${GICC_ROOT} bash omp/build_libgicc_omp.sh" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
set -x
"${CLANG}" -fopenmp --offload-arch="${ARCH}" --rocm-path="${ROCM}" \
    -O3 -std=gnu++17 \
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 \
    -DGICC_CPU_PROXY=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${ROCM}/include" \
    "${SOURCE}" -L"${LIBDIR}" -lgicc_omp \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    "${ROCM}/lib/libamdhip64.so" "${MPI}/lib/libmpi_cray.so" \
    "${OMPLIB}/libomptarget.so" -Wl,--allow-shlib-undefined \
    -Wl,-rpath,"${LIBDIR}:${LIBFAB}/lib64:${MPI}/lib:${OMPLIB}:${ROCM}/lib:${CRAYPE}" \
    -o "${OUTPUT}"
set +x
echo "BUILT: ${OUTPUT}"
