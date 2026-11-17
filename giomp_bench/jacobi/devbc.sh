#!/bin/bash
# devbc.sh OUT.ll [plugin]: the device IR of jacobi_giomp.cpp (-DNESTED) as
# the compile hands it to the device link, with or without the plugin.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
L=/shared/data1/Users/l1065028/llvm-22.1.7; G=/shared/data1/Users/l1065028/new-gicc
P=()
if [ "$2" = plugin ]; then P=(-fpass-plugin=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so); export GICC_MODE=chunk-lower GICC_META_DIR=$PWD/jacobi_npipe.meta; fi
$L/bin/clang++ -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version --offload-device-only -foffload-lto "${P[@]}" \
  -include string -std=gnu++17 -DNESTED -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 \
  -I$G -I$G/src -I$CUDA_HOME/include -isystem $MPI_ROOT/include -O3 -S -emit-llvm jacobi_giomp.cpp -o "$1"
