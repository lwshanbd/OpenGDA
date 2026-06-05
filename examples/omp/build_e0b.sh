#!/bin/bash
# examples/omp/build_e0b.sh - build the E0b isolated ring-push test.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
OMP_LIBDIR=/opt/rocm-6.4.0/lib/llvm/lib
mkdir -p "${GICC_ROOT}/build_ofi"
set -x
"${HIPCC}" -fopenmp --offload-arch=gfx90a -O3 -std=gnu++17 \
    -D__HIP_PLATFORM_AMD__=1 -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1 \
    -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal" \
    -I/opt/rocm-6.4.0/include \
    "${GICC_ROOT}/examples/omp/e0b_ring_push.cpp" \
    -o "${GICC_ROOT}/build_ofi/e0b_ring_push" \
    -Wl,-rpath,"${OMP_LIBDIR}":/opt/rocm-6.4.0/lib \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/e0b_ring_push"
