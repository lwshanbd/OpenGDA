#!/bin/bash
# build_e2.sh - E2 put+quiet backpressure/liveness: OpenMP app TU + HIP bridge + HIP proxy/runtime.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
OMP_LIBDIR=/opt/rocm-6.4.0/lib/llvm/lib

SCRATCH="${GICC_ROOT}/build_ofi/omp_e2_scratch"
mkdir -p "${SCRATCH}"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
trap 'rm -rf "${OBJ_DIR}"' EXIT

# Shared GICC defines/includes.
COMMON_DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
              -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
COMMON_INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
              -I/opt/rocm-6.4.0/include -I"${LIBFAB}/include" -isystem "${MPI}/include" )

set -x
# (1) OpenMP application TU.
"${HIPCC}" -fopenmp --offload-arch=gfx90a -O3 -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/examples/omp/e2_put_quiet_loop.cpp" -o "${OBJ_DIR}/e2.o"

# (2) HIP bridge TU.
"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/examples/omp/gicc_omp_bridge_hip.cpp" -o "${OBJ_DIR}/bridge.o"

# (3) HIP runtime + proxy TUs (same as build_xnode_proxy_bench.sh).
"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" -o "${OBJ_DIR}/runtime_helpers.o"
for src in proxy_thread.cpp proxy_libfabric.cpp; do
    "${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
        "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
        -c "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" -o "${OBJ_DIR}/${src%.cpp}.o"
done

# (4) Link: -fopenmp drives the OpenMP device link; embedded HIP fatbins from
# the bridge/runtime .o files are handled by --offload-arch. Drop --hip-link
# (that overrides the driver and drops libomptarget, causing __tgt_target_kernel
# undefined-symbol). The -L/${OMP_LIBDIR} -lomptarget makes the OpenMP target
# runtime explicit so AMDGPU offload resolves at link time.
"${HIPCC}" -fopenmp --offload-arch=gfx90a \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/e2.o" "${OBJ_DIR}/bridge.o" \
    "${OBJ_DIR}/runtime_helpers.o" "${OBJ_DIR}/proxy_thread.o" "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/e2_put_quiet_loop" \
    -L"${OMP_LIBDIR}" -lomptarget \
    -Wl,-rpath,"${LIBFAB}/lib64":"${MPI}/lib":"${OMP_LIBDIR}":/opt/rocm-6.4.0/lib \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/e2_put_quiet_loop"
