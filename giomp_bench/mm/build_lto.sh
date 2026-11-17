#!/bin/bash
# Build GiOMP MiniMod for MAPLE (GH200 + InfiniBand) with clang 22.
#
# target_3d.c is compiled as C++ so that it can include gicc/omp_pipeline.h.
# PIPE=1 builds it with put discovery: the device compile of the unit is
# run twice with the gicc-passes plugin (GICC_MODE=put-discover, then
# chunk-lower), and the plugin moves the face puts after the fused kernel
# (GICC_HALO_OVERLAP=2) into that kernel. PIPE=0 is the same source without
# the plugin.
#
# Usage: [PIPE=0|1] build_maple_ib.sh OUTPUT
#   LLVM       clang 22 install (default /shared/data1/Users/l1065028/llvm-22.1.7)
#   GICC_ROOT  GICC checkout with the IB backend
#   GIOMP_LIBDIR  directory holding libgicc_omp.so built for GIOMP_BACKEND=ib
#   GICC_PASSES_SO  libgicc-passes.so built against LLVM 22
#   MM_DEFS    extra -D flags, e.g. -DMINIMOD_FUSED_KERNEL
# Expects the Maple modules of env_cudampi.sh (CUDA_HOME, MPI_ROOT).
set -euo pipefail

OUT="${1:?usage: build_maple_ib.sh OUTPUT}"
MM=/shared/data1/Users/l1065028/minimod-giomp
LLVM="${LLVM:-/shared/data1/Users/l1065028/llvm-22.1.7}"
GICC_ROOT="${GICC_ROOT:?set GICC_ROOT}"
LIBDIR="${GIOMP_LIBDIR:?set GIOMP_LIBDIR}"
CUDA="${CUDA_HOME:?load the cuda module}"
MPI="${MPI_ROOT:?load the openmpi module}"
PIPE="${PIPE:-0}"
RT="${LLVM}/lib/aarch64-unknown-linux-gnu"
CC="${LLVM}/bin/clang"
CXX="${LLVM}/bin/clang++"

OFFLOAD=( -fopenmp --offload-arch=sm_90 --cuda-path="${CUDA}" -Wno-unknown-cuda-version )
DEFS=( -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 ${MM_DEFS:-} )
INCS=( -I"${MM}" -I"${MM}/targets" -I"${GICC_ROOT}" -I"${GICC_ROOT}/src"
       -I"${CUDA}/include" -isystem "${MPI}/include" )
LTO=( ${FORCE_LTO:+-foffload-lto} ${DIST_PLUGIN:+-fpass-plugin=${DIST_PLUGIN}} )
OBJ="$(mktemp -d)"
trap 'rm -rf "${OBJ}"' EXIT

if [[ "${PIPE}" == "1" ]]; then
    PASSES="${GICC_PASSES_SO:?set GICC_PASSES_SO}"
    export GICC_META_DIR="${GICC_META_DIR:-${OUT}.meta}"
    rm -rf "${GICC_META_DIR}"
    mkdir -p "${GICC_META_DIR}"
    LTO=( -foffload-lto -fpass-plugin="${PASSES}" )
fi

objs=()
for src in constants.c grid.c pml.c targets/omp_gicc/main_gicc.c \
           targets/omp_gicc/data_setup.c targets/omp_gicc/multi_gpu.c; do
    o="${OBJ}/$(basename "${src}" .c).o"
    "${CC}" "${OFFLOAD[@]}" "${LTO[@]/-fpass-plugin=*/}" -O3 -std=gnu11 -x c \
        "${DEFS[@]}" "${INCS[@]}" -c "${MM}/${src}" -o "${o}"
    objs+=("${o}")
done

CXXFLAGS=( "${OFFLOAD[@]}" -O3 -std=gnu++17 -include string "${DEFS[@]}" "${INCS[@]}" )
if [[ "${PIPE}" == "1" ]]; then
    # Host side only: record the puts that follow each launch.
    GICC_MODE=put-discover "${CXX}" "${CXXFLAGS[@]}" --offload-host-only \
        -fpass-plugin="${PASSES}" -x c++ -c "${MM}/targets/omp_gicc/target_3d.c" \
        -o "${OBJ}/discover.o"
fi
if [[ "${AS_C:-0}" == "1" ]]; then
    "${CC}" "${OFFLOAD[@]}" -O3 -std=gnu11 -x c "${DEFS[@]}" "${INCS[@]}" \
        -c "${MM}/targets/omp_gicc/target_3d.c" -o "${OBJ}/target_3d.o"
else
    GICC_MODE=chunk-lower "${CXX}" "${CXXFLAGS[@]}" "${LTO[@]}" -x c++ \
        -c "${MM}/targets/omp_gicc/target_3d.c" -o "${OBJ}/target_3d.o"
fi
GICC_MODE=chunk-lower "${CXX}" "${CXXFLAGS[@]}" "${LTO[@]}" \
    -c "${MM}/targets/omp_gicc/halo_dev.cpp" -o "${OBJ}/halo_dev.o"
objs+=("${OBJ}/target_3d.o" "${OBJ}/halo_dev.o")

# The device code is linked (and, with PIPE=1, lowered by the plugin) here.
GICC_MODE=chunk-lower "${CXX}" "${OFFLOAD[@]}" "${LTO[@]}" ${GIOMP_LINK_FLAGS:-} \
    "${objs[@]}" -o "${OUT}" \
    -L"${LIBDIR}" -lgicc_omp "${RT}/libomptarget.so" \
    -L"${CUDA}/lib64" -lcudart -lcuda -libverbs -lmlx5 -lpthread \
    -L"${MPI}/lib" -lmpi -lm \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,"${LIBDIR}:${RT}:${LLVM}/lib:${CUDA}/lib64:${MPI}/lib"
echo "BUILT (PIPE=${PIPE}): ${OUT}"
