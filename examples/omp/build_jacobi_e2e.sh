#!/bin/bash
# build_jacobi_e2e.sh - build BOTH transports of jacobi_e2e.cpp with the same
# ROCm 6.4.0 clang so the compute kernel is identical:
#   build_ofi/jacobi_e2e_proxy : no pass, halo puts via ompx_put_proxy
#   build_ofi/jacobi_e2e_dwq   : 2-pass GICC_MODE=omp-dwq LTO pass build; the
#                                pass synthesizes the per-launch host trace and
#                                lowers ompx_dwq_flush to the MMIO trigger.
# Both keep -foffload-lto and -openmp-opt-disable so codegen matches.
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
# Override SRC/OUT_STEM to build another single-source e2e benchmark
# (e.g. SRC=.../batch_e2e.cpp OUT_STEM=batch_e2e).
SRC="${GIOMP_E2E_SRC:-${GICC_ROOT}/examples/omp/jacobi_e2e.cpp}"
OUT_STEM="${GIOMP_E2E_OUT:-jacobi_e2e}"
META_DIR="${GICC_META_DIR:-/tmp/gicc-${OUT_STEM}-meta}"

DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
       -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
       -I"${ROCM}/include" -I"${LIBFAB}/include" -isystem "${MPI}/include" )

link_bin() {  # $1=obj $2=out
  "${CLANG}" -fopenmp --offload-arch=gfx90a \
      --rtlib=compiler-rt -unwindlib=libgcc \
      -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
      "$1" -L"${LIBDIR}" -lgicc_omp \
      "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
      "${ROCM}/lib/libamdhip64.so" "${MPI}/lib/libmpi_cray.so" "${OMPLIB}/libomptarget.so" \
      -Wl,--allow-shlib-undefined \
      -Wl,-rpath,"${LIBDIR}":"${LIBFAB}/lib64":"${MPI}/lib":"${OMPLIB}":"${ROCM}/lib":"${CRAYPE}" \
      -o "$2"
}

OBJ="$(mktemp -d)"; trap 'rm -rf "${OBJ}"' EXIT

# ---- proxy variant (no pass) ----
"${CLANG}" -fopenmp --offload-arch=gfx90a -foffload-lto \
    -mllvm -openmp-opt-disable=true -O3 -std=gnu++17 \
    "${DEFS[@]}" "${INCS[@]}" -c "${SRC}" -o "${OBJ}/proxy.o"
link_bin "${OBJ}/proxy.o" "${GICC_ROOT}/build_ofi/${OUT_STEM}_proxy"

# ---- dwq variant (2-pass compile through the LTO pass) ----
mkdir -p "${META_DIR}"; rm -f "${META_DIR}"/*.json
for pass in 1 2; do
  GICC_MODE=omp-dwq GICC_META_DIR="${META_DIR}" \
  "${CLANG}" -fopenmp --offload-arch=gfx90a -foffload-lto \
      -mllvm -openmp-opt-disable=true -fpass-plugin="${PLUGIN}" \
      -O3 -std=gnu++17 \
      "${DEFS[@]}" "${INCS[@]}" -c "${SRC}" -o "${OBJ}/dwq.o"
done
link_bin "${OBJ}/dwq.o" "${GICC_ROOT}/build_ofi/${OUT_STEM}_dwq"

echo "META_DIR JSON:"; ls "${META_DIR}"/*.json 2>/dev/null || echo "  (none)"
echo "BUILT: ${GICC_ROOT}/build_ofi/${OUT_STEM}_proxy"
echo "BUILT: ${GICC_ROOT}/build_ofi/${OUT_STEM}_dwq"
