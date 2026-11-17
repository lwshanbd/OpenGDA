#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
L=/shared/data1/Users/l1065028/llvm-22.1.7
F="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3"
for t in t_asm2 t_asm3; do
  $L/bin/clang++ $F -std=gnu++17 -c $t.cpp -o $t.o
  mkdir -p l_$t; (cd l_$t && $L/bin/clang $F -Wl,--device-linker=--save-temps ../gi_orig/t.o ../$t.o -o k.so -shared -Wl,--allow-shlib-undefined >/dev/null 2>&1)
  f=$(ls l_$t/*.s 2>/dev/null | head -1); [ -n "$f" ] || { echo "$t: no PTX"; continue; }; s=$(grep -n "^.weak .entry.*target_inner_3d.*_l147" $f | cut -d: -f1); e=$(awk -v s=$s 'NR>s && /^}/{print NR; exit}' $f)
  echo "$t: inner l147 $((e-s)) lines, malloc $(sed -n "${s},${e}p" $f | grep -c malloc)"
done
