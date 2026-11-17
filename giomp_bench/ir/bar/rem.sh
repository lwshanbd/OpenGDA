#!/bin/bash
# rem.sh: openmp-opt link-time remarks for the pipelined and baseline Jacobi.
cd "$(dirname "$0")"/st
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
R="-Wl,--save-temps"
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
L=/shared/data1/Users/l1065028/llvm-22.1.7; G=/shared/data1/Users/l1065028/new-gicc; T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=$G GIOMP_BACKEND=ib GIOMP_CXX=$L/bin/clang++ GIOMP_MPI_ROOT=$MPI_ROOT
export GIOMP_CUDA_ROOT=$CUDA_HOME GIOMP_OFFLOAD_ARCH=sm_90 GIOMP_LIBDIR=$T/lib2
export GIOMP_OMP_LIBDIR=$L/lib/aarch64-unknown-linux-gnu GICC_PASSES_SO=$P
GIOMP_EXTRA_FLAGS="$R" bash $G/examples/omp/build_giomp_after.sh $J/jacobi_giomp.cpp $PWD/jp_r > jp_rem.txt 2>&1
GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P $R" bash $G/examples/omp/build_giomp_example.sh $J/jacobi_mpi_omp.cpp $PWD/jm_r > jm_rem.txt 2>&1
