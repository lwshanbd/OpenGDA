#!/bin/bash
# dump.sh SRC OUT.ll [extra flags]: device IR (pre-link, O3) of an OpenMP TU on maple-1.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
L=/shared/data1/Users/l1065028/llvm-22.1.7
G=/shared/data1/Users/l1065028/new-gicc
SRC=$1; OUT=$2; shift 2
$L/bin/clang++ -fopenmp --offload-arch=sm_90 --offload-device-only -O3 -std=gnu++17 -S -emit-llvm \
  -I$G -I$G/src -isystem $MPI_ROOT/include -DGICC_BACKEND_IB "$@" $SRC -o $OUT 2>&1 | grep -v "unknown CUDA" | head
