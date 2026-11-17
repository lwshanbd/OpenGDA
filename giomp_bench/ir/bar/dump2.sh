#!/bin/bash
# dump2.sh: pre-link device IR (after OptimizerLast) of the pipelined and the
cd "$(dirname "$0")"
# baseline Jacobi, with the plugin, into ./jp and ./jm.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
L=/shared/data1/Users/l1065028/llvm-22.1.7
G=/shared/data1/Users/l1065028/new-gicc
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
F="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -include string -DGICC_GPU_CUDA=1 -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -I$CUDA_HOME/include --offload-device-only -foffload-lto -O3 -std=gnu++17 -c -emit-llvm -I$G -I$G/src -isystem $MPI_ROOT/include -fpass-plugin=$P"
export GICC_DIST_CHUNK=8192
rm -rf jp.meta; cp -r $J/ds/jp_w.meta jp.meta
GICC_MODE=chunk-lower GICC_META_DIR=$PWD/jp.meta $L/bin/clang++ $F $J/jacobi_giomp.cpp -o jp.bc 2>&1 | grep -E "gicc-|rror" | head
$L/bin/clang++ $F $J/jacobi_mpi_omp.cpp -o jm.bc 2>&1 | grep -E "gicc-|rror" | head
$L/bin/llvm-dis jp.bc; $L/bin/llvm-dis jm.bc
