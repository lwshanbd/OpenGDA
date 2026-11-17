#!/bin/bash
# The MPI MiniMod with target_3d.c compiled as C++.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm
MM=/shared/data1/Users/l1065028/minimod-mpi
/shared/data1/Users/l1065028/llvm-22.1.7/bin/clang++ -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3 -std=gnu++17 \
  -I$MM -I$MM/targets -isystem $MPI_ROOT/include -c tgt_cxx.cpp -o tgt_cxx.o
TGT_OBJ=$PWD/tgt_cxx.o bash build_mpi.sh $PWD/mm_mpi_cxx
