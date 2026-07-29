#!/bin/bash
# Build one ordinary OpenMP-target example against the prebuilt GiOMP library.
# Usage: build_giomp_example.sh SOURCE OUTPUT
#
# GIOMP_BACKEND selects the GPU stack, matching omp/build_libgicc_omp.sh:
#   hip  (default) - ROCm 6.4.0 clang, gfx90a (Tioga)
#   cuda           - clang with NVPTX offload, sm_90 (GH200 + Slingshot)
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SOURCE OUTPUT" >&2
    exit 2
fi

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BACKEND="${GIOMP_BACKEND:-hip}"
LIBFAB="${GIOMP_LIBFABRIC_ROOT:-/opt/cray/libfabric/2.1}"
MPI="${GIOMP_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0}"
CRAYPE="${GIOMP_CRAYPE_LIBDIR:-/opt/cray/pe/lib64}"
LIBDIR="${GIOMP_LIBDIR:-${GICC_ROOT}/build_ofi/lib}"
SOURCE="$1"
OUTPUT="$2"

if [[ ! -f "${LIBDIR}/libgicc_omp.so" ]]; then
    echo "error: ${LIBDIR}/libgicc_omp.so is missing" >&2
    echo "build it first with: GICC_ROOT=${GICC_ROOT} bash omp/build_libgicc_omp.sh" >&2
    exit 1
fi

case "${BACKEND}" in
hip)
    ROCM="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.0}"
    CLANG="${GIOMP_CXX:-${ROCM}/lib/llvm/bin/clang++}"
    ARCH="${GIOMP_OFFLOAD_ARCH:-gfx90a}"
    OMPLIB="${GIOMP_OMP_LIBDIR:-${ROCM}/lib/llvm/lib}"
    MPI_GTL="${GIOMP_MPI_GTL:-/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_hsa.so}"
    OFFLOAD=( --offload-arch="${ARCH}" --rocm-path="${ROCM}" )
    GPU_DEFS=( -DGICC_GPU_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 )
    GPU_INCS=( -I"${ROCM}/include" )
    GPU_LIBS=( "${ROCM}/lib/libamdhip64.so" )
    RPATH_EXTRA="${ROCM}/lib"
    ;;
cuda)
    CUDA="${GIOMP_CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda}}"
    CLANG="${GIOMP_CXX:?set GIOMP_CXX to a clang++ with NVPTX offload}"
    ARCH="${GIOMP_OFFLOAD_ARCH:-sm_90}"
    OMPLIB="${GIOMP_OMP_LIBDIR:?set GIOMP_OMP_LIBDIR to the clang lib dir}"
    MPI_GTL="${GIOMP_MPI_GTL:-/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_cuda.so}"
    # -include string: CUDA's host_defines.h does `#define __noinline__
    # __attribute__((noinline))`, which breaks libstdc++ 13+'s use of
    # [[__gnu__::__noinline__]] in <string>. Pulling <string> in before any
    # CUDA header sidesteps it; without this, any example that uses
    # std::string fails to parse in the NVPTX device pass.
    OFFLOAD=( --offload-arch="${ARCH}" --cuda-path="${CUDA}"
              -Wno-unknown-cuda-version -include string )
    GPU_DEFS=( -DGICC_GPU_CUDA=1 )
    GPU_INCS=( -I"${CUDA}/include" )
    GPU_LIBS=( "-L${CUDA}/lib64" -lcudart )
    RPATH_EXTRA="${CUDA}/lib64"
    ;;
*)
    echo "error: GIOMP_BACKEND must be 'hip' or 'cuda' (got '${BACKEND}')" >&2
    exit 2
    ;;
esac

# Cray MPICH names its library after the compiler that built it
# (libmpi_cray.so, libmpi_gnu_123.so, ...), so probe rather than hard-code.
MPI_LIB="${GIOMP_MPI_LIB:-}"
if [[ -z "${MPI_LIB}" ]]; then
    for cand in "${MPI}/lib/libmpi_cray.so" "${MPI}"/lib/libmpi_gnu_*.so \
                "${MPI}/lib/libmpich.so"; do
        [[ -f "${cand}" ]] && { MPI_LIB="${cand}"; break; }
    done
fi
if [[ -z "${MPI_LIB}" ]]; then
    echo "error: no Cray MPICH library found under ${MPI}/lib" >&2
    exit 1
fi

# Extra flags for sources that select a backend at compile time (e.g. the
# jacobi / mm evals, which build the same file as GiOMP or as plain MPI).
read -r -a EXTRA_FLAGS <<< "${GIOMP_EXTRA_FLAGS:-}"

mkdir -p "$(dirname "${OUTPUT}")"
set -x
"${CLANG}" -fopenmp "${OFFLOAD[@]}" \
    -O3 -std=gnu++17 \
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_CPU_PROXY=1 \
    "${GPU_DEFS[@]}" "${EXTRA_FLAGS[@]}" \
    -I"${GICC_ROOT}" -I"${GICC_ROOT}/src" "${GPU_INCS[@]}" \
    -isystem "${MPI}/include" \
    "${SOURCE}" -L"${LIBDIR}" -lgicc_omp \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    "${GPU_LIBS[@]}" "${MPI_LIB}" "${MPI_GTL}" \
    "${OMPLIB}/libomptarget.so" -Wl,--allow-shlib-undefined \
    -Wl,-rpath,"${LIBDIR}:${LIBFAB}/lib64:${MPI}/lib:$(dirname "${MPI_GTL}"):${OMPLIB}:${RPATH_EXTRA}:${CRAYPE}" \
    -o "${OUTPUT}"
set +x
echo "BUILT (${BACKEND}): ${OUTPUT}"
