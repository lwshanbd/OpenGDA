#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
MM=/shared/data1/Users/l1065028/minimod-giomp; G=/shared/data1/Users/l1065028/new-gicc
L=/shared/data1/Users/l1065028/llvm-22.1.7
F="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3"
I="-DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include -I$MM -I$MM/targets -isystem $MPI_ROOT/include"
for p in ${PROBES:-p_noput p_put p_quiet}; do
  $L/bin/clang++ $F -std=gnu++17 -include string $I -c $p.cpp -o $p.o 2>&1 | grep error
  d=l_$p; mkdir -p $d; (cd $d && $L/bin/clang $F -Wl,--device-linker=--save-temps ../t_new.o ../$p.o -o k.so -shared -Wl,--allow-shlib-undefined >/dev/null 2>&1)
  f=$(ls -t $d/*.s 2>/dev/null | head -1); [ -n "$f" ] || { echo "$p: no PTX"; continue; }
  s=$(grep -n "^.weak .entry.*target_inner_3d.*_l147" $f | cut -d: -f1); e=$(awk -v s=$s 'NR>s && /^}/{print NR; exit}' $f)
  echo "$p: inner l147 $((e-s)) lines, malloc $(sed -n "${s},${e}p" $f | grep -c malloc)"
done
