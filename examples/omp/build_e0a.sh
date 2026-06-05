#!/bin/bash
# examples/omp/build_e0a.sh
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
OMP_LIBDIR=/opt/rocm-6.4.0/lib/llvm/bin/../lib
HIP_LIBDIR=/opt/rocm-6.4.0/lib
mkdir -p "${GICC_ROOT}/build_ofi"
set -x
"${HIPCC}" -fopenmp --offload-arch=gfx90a -O3 -std=gnu++17 \
    -D__HIP_PLATFORM_AMD__=1 \
    -I/opt/rocm-6.4.0/include \
    "${GICC_ROOT}/examples/omp/e0a_atomic_interop.cpp" \
    -o "${GICC_ROOT}/build_ofi/e0a_atomic_interop" \
    -Wl,-rpath,"${OMP_LIBDIR}" \
    -Wl,-rpath,"${HIP_LIBDIR}" \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/e0a_atomic_interop"
