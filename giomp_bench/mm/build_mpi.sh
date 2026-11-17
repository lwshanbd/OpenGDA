#!/bin/bash
# build_mpi.sh OUT: the OpenMP offload + CUDA-aware MPI MiniMod
# (minimod-mpi, omp_offload_opt-gpu_multi_nowait) for Maple, with the
# toolchain and flags of the GiOMP builds (clang 22, sm_90, -foffload-lto).
set -euo pipefail
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
OUT="${1:?usage: build_mpi.sh OUT}"
MM=/shared/data1/Users/l1065028/minimod-mpi
LLVM=/shared/data1/Users/l1065028/llvm-22.1.7
RT="${LLVM}/lib/aarch64-unknown-linux-gnu"
T="${MM}/targets/omp_offload_opt-gpu_multi_nowait"
"${LLVM}/bin/clang" -fopenmp --offload-arch=sm_90 --cuda-path="${CUDA_HOME}" \
    -Wno-unknown-cuda-version -Wno-error=incompatible-pointer-types -foffload-lto ${MM_DEFS:-} ${DIST_PLUGIN:+-fpass-plugin=${DIST_PLUGIN}} -O3 -std=gnu11 -x c \
    -I"${MM}" -I"${MM}/targets" -isystem "${MPI_ROOT}/include" \
    "${MM}/constants.c" "${MM}/grid.c" "${MM}/main.c" "${MM}/pml.c" \
    "${T}/data_setup.c" "${T}/multi_gpu.c" ${TGT_OBJ:+-x none "${TGT_OBJ}"} \
    $([ -n "${TGT_OBJ:-}" ] || echo "${T}/target_3d.c") -o "${OUT}" \
    -L"${MPI_ROOT}/lib" -lmpi -lm ${TGT_OBJ:+-lstdc++} -Wl,-rpath,"${RT}:${LLVM}/lib:${MPI_ROOT}/lib"
echo "BUILT: ${OUT}"
