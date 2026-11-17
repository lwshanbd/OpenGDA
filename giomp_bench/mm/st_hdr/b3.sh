#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
MM=/shared/data1/Users/l1065028/minimod-giomp; G=/shared/data1/Users/l1065028/new-gicc
L=/shared/data1/Users/l1065028/llvm-22.1.7
F="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3"
I="-DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include -I$MM -I$MM/targets -isystem $MPI_ROOT/include"
$L/bin/clang++ $F -std=gnu++17 -include string $I -c $MM/targets/omp_gicc/halo_dev.cpp -o halo.o
for s in data_setup multi_gpu main_gicc; do $L/bin/clang $F -std=gnu11 -x c $I -c $MM/targets/omp_gicc/$s.c -o $s.o; done
try() { name=$1; shift; mkdir -p $name; (cd $name && $L/bin/clang $F -Wl,--device-linker=--save-temps "$@" -o k.so -shared -Wl,--allow-shlib-undefined >/dev/null 2>&1)
  echo "== $name: $(/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump -res-usage $name/*.cubin | grep -A1 'Function __omp' | paste - - | grep -E 'inner|pml' | sed -E 's/.*(target_[a-z]+_3d).*(_l[0-9]+):.*REG:([0-9]+).*/\1\2=\3/' | tr '\n' ' ')"; }

$L/bin/clang++ $F -std=gnu++17 -include string $I -c halo_spmd.cpp -o halo_spmd.o


F0="$F"; F="$F -fopenmp-assume-no-thread-state"
$L/bin/clang++ $F -std=gnu++17 -include string $I -c halo_dev_copy.cpp -o halo_nts.o 2>/dev/null || $L/bin/clang++ $F -std=gnu++17 -include string $I -c $MM/targets/omp_gicc/halo_dev.cpp -o halo_nts.o
try l_nts ../gi_orig/t.o ../halo_nts.o
