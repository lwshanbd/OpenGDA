#!/bin/bash
# build_libgicc_omp.sh - produce libgicc_omp.{so,a} from the -x hip runtime TUs
# + ompx_host.cpp, using the ROCm 6.4.0 clang. This is the SAME toolchain the
# GICC LTO pass plugin (libgicc-passes.so) is built against, so the DWQ path
# (app TU compiled -fpass-plugin) links this library cleanly -- a clang-21
# build could not, because the LLVM-19 pass plugin will not load in clang-21.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-/p/lustre2/shan4/new-gicc}"
ROCM="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.0}"
HIPCC="${GIOMP_CXX:-${ROCM}/lib/llvm/bin/clang++}"
LIBFAB="${GIOMP_LIBFABRIC_ROOT:-/opt/cray/libfabric/2.1}"
MPI="${GIOMP_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0}"
ARCH="${GIOMP_OFFLOAD_ARCH:-gfx90a}"
BUILD_SHARED="${GIOMP_BUILD_SHARED:-1}"
OUTDIR="${GIOMP_OUTDIR:-${GICC_ROOT}/build_ofi/lib}"; mkdir -p "${OUTDIR}"
OBJ="$(mktemp -d)"; trap 'rm -rf "${OBJ}"' EXIT

DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DGICC_CPU_PROXY=1
       -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
       -I"${ROCM}/include" -I"${LIBFAB}/include" -isystem "${MPI}/include" )

SRCS=(
  "${GICC_ROOT}/src/gicc/omp/ompx_host.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_thread.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"
)
OBJS=()
for s in "${SRCS[@]}"; do
  o="${OBJ}/$(basename "${s%.cpp}").o"
  "${HIPCC}" -x hip -O3 -fPIC --offload-arch="${ARCH}" --rocm-path="${ROCM}" -std=gnu++17 -fopenmp \
      "${DEFS[@]}" "${INCS[@]}" -c "${s}" -o "${o}"
  OBJS+=("${o}")
done

# static
ar rcs "${OUTDIR}/libgicc_omp.a" "${OBJS[@]}"
if [[ "${BUILD_SHARED}" != "0" ]]; then
  "${HIPCC}" -shared -fPIC --offload-arch="${ARCH}" --rocm-path="${ROCM}" \
      "${OBJS[@]}" -o "${OUTDIR}/libgicc_omp.so"
  echo "BUILT: ${OUTDIR}/libgicc_omp.so ${OUTDIR}/libgicc_omp.a"
else
  echo "BUILT: ${OUTDIR}/libgicc_omp.a"
fi
