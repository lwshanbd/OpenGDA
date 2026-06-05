#!/bin/bash
# build_e_ompmap.sh - make-or-break spike: register an OpenMP target-mapped
# buffer with GICC and put one face. Same recipe as build_e1.sh, app source
# swapped to e_omp_mapped_put.cpp (proxy path, no LTO pass).
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
OMP_LIBDIR=/opt/rocm-6.4.0/lib/llvm/lib

SCRATCH="${GICC_ROOT}/build_ofi/omp_ompmap_scratch"
mkdir -p "${SCRATCH}"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
trap 'rm -rf "${OBJ_DIR}"' EXIT

COMMON_DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
              -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
COMMON_INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
              -I/opt/rocm-6.4.0/include -I"${LIBFAB}/include" -isystem "${MPI}/include" )

set -x
"${HIPCC}" -fopenmp --offload-arch=gfx90a -O3 -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/examples/omp/e_omp_mapped_put.cpp" -o "${OBJ_DIR}/app.o"

"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/examples/omp/gicc_omp_bridge_hip.cpp" -o "${OBJ_DIR}/bridge.o"

"${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
    "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
    -c "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" -o "${OBJ_DIR}/runtime_helpers.o"
for src in proxy_thread.cpp proxy_libfabric.cpp; do
    "${HIPCC}" -x hip -O3 --offload-arch=gfx90a -std=gnu++17 \
        "${COMMON_DEFS[@]}" "${COMMON_INCS[@]}" \
        -c "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" -o "${OBJ_DIR}/${src%.cpp}.o"
done

"${HIPCC}" -fopenmp --offload-arch=gfx90a \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/app.o" "${OBJ_DIR}/bridge.o" \
    "${OBJ_DIR}/runtime_helpers.o" "${OBJ_DIR}/proxy_thread.o" "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/e_omp_mapped_put" \
    -L"${OMP_LIBDIR}" -lomptarget \
    -Wl,-rpath,"${LIBFAB}/lib64":"${MPI}/lib":"${OMP_LIBDIR}":/opt/rocm-6.4.0/lib \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/e_omp_mapped_put"
