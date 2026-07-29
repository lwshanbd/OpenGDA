#!/bin/bash
# build_libgicc_omp.sh - produce libgicc_omp.{so,a} from the GPU runtime TUs
# + ompx_host.cpp.
#
# Two backends, selected by GIOMP_BACKEND:
#
#   hip  (default) - ROCm 6.4.0 clang, AMDGCN. This is the SAME toolchain the
#                    GICC LTO pass plugin (libgicc-passes.so) is built against,
#                    so the DWQ path (app TU compiled -fpass-plugin) links this
#                    library cleanly -- a clang-21 build could not, because the
#                    LLVM-19 pass plugin will not load in clang-21.
#   cuda           - upstream clang with NVPTX offload, for GH200 + Slingshot.
#                    The pass plugin is NOT available there, so this covers the
#                    IPC / CPU-proxy transports only.
#
# The TUs are compiled as GPU source (-x hip / -x cuda) rather than plain C++
# because ofi_runtime.hpp transitively pulls fabric.hpp, which defines a
# __global__ trigger kernel.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-/p/lustre2/shan4/new-gicc}"
BACKEND="${GIOMP_BACKEND:-hip}"
LIBFAB="${GIOMP_LIBFABRIC_ROOT:-/opt/cray/libfabric/2.1}"
MPI="${GIOMP_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0}"
BUILD_SHARED="${GIOMP_BUILD_SHARED:-1}"
OUTDIR="${GIOMP_OUTDIR:-${GICC_ROOT}/build_ofi/lib}"; mkdir -p "${OUTDIR}"
OBJ="$(mktemp -d)"; trap 'rm -rf "${OBJ}"' EXIT

COMMON_DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_CPU_PROXY=1 )
COMMON_INCS=( -I"${GICC_ROOT}" -I"${GICC_ROOT}/src"
              -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
              -I"${LIBFAB}/include" -isystem "${MPI}/include" )

case "${BACKEND}" in
hip)
    ROCM="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.0}"
    CXX="${GIOMP_CXX:-${ROCM}/lib/llvm/bin/clang++}"
    ARCH="${GIOMP_OFFLOAD_ARCH:-gfx90a}"
    GPU_LANG=( -x hip --offload-arch="${ARCH}" --rocm-path="${ROCM}" )
    GPU_DEFS=( -DGICC_GPU_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
    GPU_INCS=( -I"${ROCM}/include" )
    LINK_EXTRA=( --offload-arch="${ARCH}" --rocm-path="${ROCM}" )
    ;;
cuda)
    CUDA="${GIOMP_CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda}}"
    CXX="${GIOMP_CXX:?set GIOMP_CXX to a clang++ with NVPTX offload}"
    ARCH="${GIOMP_OFFLOAD_ARCH:-sm_90}"
    GPU_LANG=( -x cuda --offload-arch="${ARCH}" --cuda-path="${CUDA}"
               -Wno-unknown-cuda-version )
    GPU_DEFS=( -DGICC_GPU_CUDA=1 )
    GPU_INCS=( -I"${CUDA}/include" )
    LINK_EXTRA=( --cuda-path="${CUDA}" )
    ;;
*)
    echo "error: GIOMP_BACKEND must be 'hip' or 'cuda' (got '${BACKEND}')" >&2
    exit 2
    ;;
esac

SRCS=(
  "${GICC_ROOT}/src/gicc/omp/ompx_host.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_thread.cpp"
  "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"
)
OBJS=()
for s in "${SRCS[@]}"; do
  o="${OBJ}/$(basename "${s%.cpp}").o"
  "${CXX}" "${GPU_LANG[@]}" -O3 -fPIC -std=gnu++17 -fopenmp \
      "${COMMON_DEFS[@]}" "${GPU_DEFS[@]}" "${COMMON_INCS[@]}" "${GPU_INCS[@]}" \
      -c "${s}" -o "${o}"
  OBJS+=("${o}")
done

# static
ar rcs "${OUTDIR}/libgicc_omp.a" "${OBJS[@]}"
if [[ "${BUILD_SHARED}" != "0" ]]; then
  "${CXX}" -shared -fPIC "${LINK_EXTRA[@]}" \
      "${OBJS[@]}" -o "${OUTDIR}/libgicc_omp.so"
  echo "BUILT (${BACKEND}): ${OUTDIR}/libgicc_omp.so ${OUTDIR}/libgicc_omp.a"
else
  echo "BUILT (${BACKEND}): ${OUTDIR}/libgicc_omp.a"
fi
