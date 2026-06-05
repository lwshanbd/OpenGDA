#!/bin/bash
# build_halo_dwq.sh - DWQ ring halo from omp target region (Phase-2 M5).
# Same 2-pass plugin build as build_e_dwq.sh; only the app source + output differ.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
OMP_LIBDIR=/opt/rocm-6.4.0/lib/llvm/lib
PLUGIN="${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so"
META_DIR="${GICC_META_DIR:-/tmp/gicc-omp-halo-dwq-meta}"

if [[ ! -f "${PLUGIN}" ]]; then
    echo "error: libgicc-passes.so not built at ${PLUGIN}" >&2
    exit 1
fi

SCRATCH="${GICC_ROOT}/build_ofi/omp_halo_dwq_scratch"
mkdir -p "${SCRATCH}"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
trap 'rm -rf "${OBJ_DIR}"' EXIT

# Fresh meta dir so a stale JSON from another kernel never leaks in.
mkdir -p "${META_DIR}"
rm -f "${META_DIR}"/*.json

# Shared GICC defines/includes.
COMMON_DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
              -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
COMMON_INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
              -I/opt/rocm-6.4.0/include -I"${LIBFAB}/include" -isystem "${MPI}/include" )

set -x
# (1) OpenMP application TU, compiled WITH the pass plugin in omp-dwq mode.
#     Compiled TWICE with the same META_DIR for cross-module JSON consistency
#     (pass 1 populates the per-kernel JSON, pass 2 consumes it).
#
#     CRITICAL: -mllvm -openmp-opt-disable=true. Without it, OpenMPOpt's
#     device-side internalization + Attributor argument-specialization folds
#     the put marker's compile-time-constant operands away from the call site,
#     so the device discovery pass loses size/offsets and the host trace would
#     enqueue zero-byte DWQ writes. Disabling OpenMPOpt preserves the full
#     7-arg marker call so every put operand survives to the JSON.
OMPOPT_DISABLE=( -mllvm -openmp-opt-disable=true )
for pass in 1 2; do
    GICC_MODE=omp-dwq GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" -fopenmp --offload-arch=gfx90a -foffload-lto \
        "${OMPOPT_DISABLE[@]}" \
        -fpass-plugin="${PLUGIN}" \
        -O3 -std=gnu++17 \
        "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
        -c "${GICC_ROOT}/examples/omp/halo_dwq.cpp" -o "${OBJ_DIR}/halo_dwq.o"
done

# (2) HIP bridge TU.
"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/examples/omp/gicc_omp_bridge_hip.cpp" -o "${OBJ_DIR}/bridge.o"

# (3) HIP runtime + proxy TUs. runtime_helpers.cpp provides
#     gicc_runtime_dwq_enqueue (the symbol the host trace calls).
"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" -o "${OBJ_DIR}/runtime_helpers.o"
for src in proxy_thread.cpp proxy_libfabric.cpp; do
    "${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
        "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
        -c "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" -o "${OBJ_DIR}/${src%.cpp}.o"
done

# (4) Link: same recipe as build_e_dwq.sh (NO --hip-link; explicit -lomptarget).
"${HIPCC}" -fopenmp --offload-arch=gfx90a \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/halo_dwq.o" "${OBJ_DIR}/bridge.o" \
    "${OBJ_DIR}/runtime_helpers.o" "${OBJ_DIR}/proxy_thread.o" "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/halo_dwq" \
    -L"${OMP_LIBDIR}" -lomptarget \
    -Wl,-rpath,"${LIBFAB}/lib64":"${MPI}/lib":"${OMP_LIBDIR}":/opt/rocm-6.4.0/lib \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
set +x
echo "META_DIR JSON:"; ls -la "${META_DIR}"/*.json || echo "  (none)"
echo "BUILT: ${GICC_ROOT}/build_ofi/halo_dwq"
