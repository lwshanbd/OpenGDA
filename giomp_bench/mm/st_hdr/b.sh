#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
MM=/shared/data1/Users/l1065028/minimod-mpi; G=/shared/data1/Users/l1065028/new-gicc
L=/shared/data1/Users/l1065028/llvm-22.1.7
$L/bin/clang++ -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3 -std=gnu++17 -include string \
  -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include \
  -I$MM -I$MM/targets -isystem $MPI_ROOT/include -c tgt.cpp -o tgt.o 2>&1 | grep -E " error" 
$L/bin/clang -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -Wl,--device-linker=--save-temps tgt.o -o k.so -shared -Wl,--allow-shlib-undefined 2>&1 | grep -E " error"
ls
