#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
MM=/shared/data1/Users/l1065028/minimod-giomp; G=/shared/data1/Users/l1065028/new-gicc
L=/shared/data1/Users/l1065028/llvm-22.1.7
for v in gi_orig gi_one; do
  mkdir -p $v
  $L/bin/clang++ -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3 -std=gnu++17 -include string \
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include \
    -I$MM -I$MM/targets -I$MM/targets/omp_gicc -isystem $MPI_ROOT/include -c $v.cpp -o $v/t.o 2>&1 | grep -E " error"
  (cd $v && $L/bin/clang -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -Wl,--device-linker=--save-temps t.o -o k.so -shared -Wl,--allow-shlib-undefined 2>&1 | grep -E " error")
  echo "== $v"; /hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump -res-usage $v/*.cubin | grep -A1 "Function __omp" | paste - - | grep -E "inner|pml" | sed -E 's/.*(target_[a-z_0-9]+).*(_l[0-9]+):.*REG:([0-9]+).*/\1 \2 REG=\3/'
done
