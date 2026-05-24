#!/bin/bash
# Plain hipcc build (no LTO pass) for asf_pattern_bench.cpp.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd ../.. && pwd)}"
OUT="${GICC_ROOT}/build_ofi/asf_pattern_bench"
mkdir -p "$(dirname $OUT)"

HIPCC=/opt/rocm-6.4.0/bin/hipcc
LIBFAB=/opt/cray/libfabric/2.1
MPICH=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0

INC=(
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I"${LIBFAB}/include"
    -isystem "${MPICH}/include"
)
DEF=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
    -DGICC_CPU_PROXY=1
)
SRC=(
    asf_pattern_bench.cpp
    "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp"
    "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"
    "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_thread.cpp"
)
LD=(
    -L"${LIBFAB}/lib64" -lfabric
    -L"${MPICH}/lib" -lmpi
    -lhwloc
    -pthread
)

set -x
$HIPCC -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${INC[@]}" "${DEF[@]}" "${SRC[@]}" "${LD[@]}" -o $OUT
echo "Built: $OUT"
