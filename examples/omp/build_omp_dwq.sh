#!/bin/bash
# build_omp_dwq.sh - build the DWQ GiOMP example (e_dwq_single) on the public
# ompx_dwq_* API. Only the app TU needs the LTO pass; the runtime lives in the
# prebuilt libgicc_omp (built by omp/build_libgicc_omp.sh with the SAME ROCm
# 6.4.0 clang, so the app TU -- compiled -fpass-plugin -- links it cleanly).
#
# 2-PASS: the host pass may run before the device pass has populated the
# per-kernel JSON on the first compile, so the app TU is compiled TWICE with the
# same GICC_META_DIR (pass 1 populates, pass 2 consumes).
#
# -openmp-opt-disable is CRITICAL: without it OpenMPOpt argument-specializes the
# put marker's constant operands away and the host trace would enqueue a
# zero-byte DWQ write.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-/p/lustre2/shan4/new-gicc}"
CLANG=/opt/rocm-6.4.0/lib/llvm/bin/clang++
ROCM=/opt/rocm-6.4.0
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
OMPLIB=/opt/rocm-6.4.0/lib/llvm/lib
CRAYPE=/opt/cray/pe/lib64
LIBDIR="${GICC_ROOT}/build_ofi/lib"
PLUGIN="${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so"
META_DIR="${GICC_META_DIR:-/tmp/gicc-omp-dwq-meta}"
OUT="${GICC_ROOT}/build_ofi/omp_dwq"

if [[ ! -f "${PLUGIN}" ]]; then echo "error: pass plugin not built at ${PLUGIN}" >&2; exit 1; fi
if [[ ! -f "${LIBDIR}/libgicc_omp.so" ]]; then
    echo "error: libgicc_omp not built; run omp/build_libgicc_omp.sh first" >&2; exit 1; fi

mkdir -p "${META_DIR}"; rm -f "${META_DIR}"/*.json
OBJ="$(mktemp -d)"; trap 'rm -rf "${OBJ}"' EXIT

DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
       -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
       -I"${ROCM}/include" -I"${LIBFAB}/include" -isystem "${MPI}/include" )

set -x
# (1) app TU, twice, with the pass in omp-dwq mode.
for pass in 1 2; do
  GICC_MODE=omp-dwq GICC_META_DIR="${META_DIR}" \
  "${CLANG}" -fopenmp --offload-arch=gfx90a -foffload-lto \
      -mllvm -openmp-opt-disable=true -fpass-plugin="${PLUGIN}" \
      -DGIOMP_ENABLE_DWQ -O3 -std=gnu++17 \
      "${DEFS[@]}" "${INCS[@]}" \
      -c "${GICC_ROOT}/examples/omp/e_dwq_single.cpp" -o "${OBJ}/e_dwq.o"
done

# (2) link against the prebuilt libgicc_omp (runtime + proxy + ompx_* host).
"${CLANG}" -fopenmp --offload-arch=gfx90a \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ}/e_dwq.o" \
    -L"${LIBDIR}" -lgicc_omp \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    "${ROCM}/lib/libamdhip64.so" "${MPI}/lib/libmpi_cray.so" "${OMPLIB}/libomptarget.so" \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,"${LIBDIR}":"${LIBFAB}/lib64":"${MPI}/lib":"${OMPLIB}":"${ROCM}/lib":"${CRAYPE}" \
    -o "${OUT}"
set +x
echo "META_DIR JSON:"; ls -la "${META_DIR}"/*.json 2>/dev/null || echo "  (none)"
echo "BUILT: ${OUT}"
