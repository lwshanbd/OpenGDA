#!/bin/bash
# Build fence_scope_sweep microbench. No LTO pass needed (direct device-side
# IPC copy, no put_no_db lowering). Just needs GICC_CPU_PROXY runtime +
# libfabric + MPI.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0

SCRATCH="${GICC_ROOT}/build_ofi/ipc_copy_scratch"
mkdir -p "${SCRATCH}"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
trap 'rm -rf "${OBJ_DIR}"' EXIT

CFLAGS=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
    -DGICC_CPU_PROXY=1
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I"${LIBFAB}/include"
    -isystem "${MPI}/include"
    -O3 --offload-arch=gfx90a -std=gnu++17
)

set -x
"${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/examples/proxy/fence_scope_sweep.cpp" \
    -o "${OBJ_DIR}/bench.o"

"${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
    -o "${OBJ_DIR}/runtime_helpers.o"

for src in proxy_thread.cpp proxy_libfabric.cpp; do
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
        "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" \
        -o "${OBJ_DIR}/${src%.cpp}.o"
done

"${HIPCC}" -O3 --offload-arch=gfx90a --hip-link \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/bench.o" \
    "${OBJ_DIR}/runtime_helpers.o" \
    "${OBJ_DIR}/proxy_thread.o" \
    "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/fence_scope_sweep" \
    -Wl,-rpath,"${LIBFAB}/lib64":"${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/fence_scope_sweep"
