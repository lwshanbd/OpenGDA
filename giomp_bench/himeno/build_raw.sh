#!/bin/bash
# Usage: build.sh SRC OUT  (on maple-1). Prints the plugin's verdicts.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
L=/shared/data1/Users/l1065028/llvm-22.1.7
G=/shared/data1/Users/l1065028/new-gicc
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=$G GIOMP_BACKEND=ib GIOMP_CXX=$L/bin/clang++ GIOMP_MPI_ROOT=$MPI_ROOT
export GIOMP_CUDA_ROOT=$CUDA_HOME GIOMP_OFFLOAD_ARCH=sm_90 GIOMP_LIBDIR=${LIBDIR:-$T/lib2}
export GIOMP_OMP_LIBDIR=$L/lib/aarch64-unknown-linux-gnu
export GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
if [ "${PLAIN:-0}" = 1 ]; then B=build_giomp_example.sh; else B=build_giomp_after.sh; fi
bash $G/examples/omp/$B "$1" "$2" 2>&1 
echo "$( [ -x $2 ] && echo BUILT || echo FAILED ) $2"
