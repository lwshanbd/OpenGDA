#!/bin/bash
# Build one Jacobi source as GiOMP, DiOMP, and GPU-aware MPI.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DIOMP_ROOT="${DIOMP_ROOT:-/p/lustre2/shan4/DiOMP}"
DIOMP_INSTALL="${DIOMP_INSTALL:-/p/lustre2/shan4/softwares/diomp}"
GASNET_ROOT="${GASNET_ROOT:-/p/lustre2/shan4/softwares/gasnet-amd}"
ROCM="${GIOMP_ROCM_ROOT:-/opt/rocm-6.4.3}"
CXX="${GIOMP_CXX:-${DIOMP_INSTALL}/bin/clang++}"
MPI="${GIOMP_MPI_ROOT:-/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0}"
MPI_GTL="${GIOMP_MPI_GTL:-/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_hsa.so}"
DIOMP_MPI="${DIOMP_MPI_ROOT:-/opt/cray/pe/mpich/8.1.31/ofi/crayclang/18.0}"
DIOMP_MPI_GTL="${DIOMP_MPI_GTL:-/opt/cray/pe/mpich/8.1.21/gtl/lib/libmpi_gtl_hsa.so}"
LIBFAB="${GIOMP_LIBFABRIC_ROOT:-/opt/cray/libfabric/2.1}"
OMP_LIB="${GIOMP_OMP_LIBDIR:-${DIOMP_INSTALL}/lib/x86_64-unknown-linux-gnu}"
CRAYPE="${GIOMP_CRAYPE_LIBDIR:-/opt/cray/pe/lib64}"
GIOMP_LIB="${GIOMP_JACOBI_LIBDIR:-${GICC_ROOT}/build_ofi/lib-clang21-jacobi}"
OUT="${GIOMP_JACOBI_OUTDIR:-${GICC_ROOT}/build_ofi/jacobi_eval}"
SRC="${GICC_ROOT}/examples/omp/giomp_jacobi_eval.cpp"

mkdir -p "${OUT}" "${GIOMP_LIB}"
export LD_LIBRARY_PATH="${DIOMP_INSTALL}/lib:${OMP_LIB}:${ROCM}/lib:${CRAYPE}:${LD_LIBRARY_PATH:-}"

GICC_ROOT="${GICC_ROOT}" GIOMP_CXX="${CXX}" GIOMP_ROCM_ROOT="${ROCM}" \
GIOMP_OMP_LIBDIR="${OMP_LIB}" GIOMP_MPI_ROOT="${MPI}" \
GIOMP_MPI_GTL="${MPI_GTL}" GIOMP_OUTDIR="${GIOMP_LIB}" \
    bash "${GICC_ROOT}/omp/build_libgicc_omp.sh"

GICC_ROOT="${GICC_ROOT}" GIOMP_CXX="${CXX}" GIOMP_ROCM_ROOT="${ROCM}" \
GIOMP_OMP_LIBDIR="${OMP_LIB}" GIOMP_MPI_ROOT="${MPI}" \
GIOMP_MPI_GTL="${MPI_GTL}" GIOMP_LIBDIR="${GIOMP_LIB}" \
    bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
        "${SRC}" "${OUT}/giomp_jacobi_eval"

COMMON=(-O3 -std=gnu++17 -fopenmp --offload-arch=gfx90a
        --rocm-path="${ROCM}" -Wno-pass-failed -I"${ROCM}/include"
        -isystem "${MPI}/include")
COMMON_LINK=("${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread
        "${ROCM}/lib/libamdhip64.so" "${MPI}/lib/libmpi_cray.so" "${MPI_GTL}"
        "${OMP_LIB}/libomptarget.so" -Wl,--allow-shlib-undefined
        -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib:$(dirname "${MPI_GTL}"):${OMP_LIB}:${ROCM}/lib:${CRAYPE}")

set -x
"${CXX}" "${COMMON[@]}" -DJACOBI_BACKEND_MPI=1 "${SRC}" \
    "${COMMON_LINK[@]}" -o "${OUT}/mpi_jacobi_eval"

"${CXX}" -O3 -std=gnu++17 -fopenmp --offload-arch=gfx90a \
    --rocm-path="${ROCM}" -Wno-pass-failed \
    -DJACOBI_BACKEND_DIOMP=1 -DDIOMP_ENABLE_HIP -D__HIP_PLATFORM_AMD__ \
    -I"${ROCM}/include" -I"${DIOMP_MPI}/include" \
    -I"${DIOMP_ROOT}/openmp/diomp/include" \
    -I"${GASNET_ROOT}/include" -I"${GASNET_ROOT}/include/ofi-conduit" \
    "${SRC}" -L"${DIOMP_INSTALL}/lib/x86_64-unknown-linux-gnu" \
    -L"${DIOMP_INSTALL}/lib" -ldiomp "${ROCM}/lib/librccl.so" \
    -L"${GASNET_ROOT}/lib" -lgasnet-ofi-par -L"${ROCM}/lib" -lamdhip64 \
    -L"${DIOMP_MPI}/lib" -lmpi "${DIOMP_MPI_GTL}" \
    -lhwloc -lrt -lm -pthread -no-pie \
    -Wl,-rpath,"${DIOMP_INSTALL}/lib:${DIOMP_INSTALL}/lib/x86_64-unknown-linux-gnu:${ROCM}/lib:${GASNET_ROOT}/lib" \
    -o "${OUT}/diomp_jacobi_eval"
set +x

echo "BUILT: ${OUT}/giomp_jacobi_eval"
echo "BUILT: ${OUT}/diomp_jacobi_eval"
echo "BUILT: ${OUT}/mpi_jacobi_eval"
