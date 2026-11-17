#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_hdr
MM=/shared/data1/Users/l1065028/minimod-giomp; G=/shared/data1/Users/l1065028/new-gicc
L=/shared/data1/Users/l1065028/llvm-22.1.7
F="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -foffload-lto -O3"
I="-DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include -I$MM -I$MM/targets -isystem $MPI_ROOT/include"
$L/bin/clang++ $F -std=gnu++17 -include string $I -c $MM/targets/omp_gicc/halo_dev.cpp -o halo_new.o 2>&1 | grep -E "error" 
$L/bin/clang++ $F -std=gnu++17 -include string $I -I$MM/targets/omp_gicc -c gi_orig.cpp -o t_new.o 2>&1 | grep -E "error"
mkdir -p l_new; (cd l_new && $L/bin/clang $F -Wl,--device-linker=--save-temps ../t_new.o ../halo_new.o -o k.so -shared -Wl,--allow-shlib-undefined >/dev/null 2>&1)
f=$(ls -t l_new/*.s 2>/dev/null | head -1); [ -n "$f" ] || { echo "no PTX"; exit 1; }
s=$(grep -n "^.weak .entry.*target_inner_3d.*_l147" $f | cut -d: -f1); e=$(awk -v s=$s 'NR>s && /^}/{print NR; exit}' $f)
echo "inner l147: $((e-s)) lines, malloc $(sed -n "${s},${e}p" $f | grep -c malloc)"
/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump -res-usage $(ls -t l_new/*.cubin | head -1) | grep -A1 "Function __omp" | paste - - | grep -E "inner|pml|halo" | sed -E 's/.*(target_[a-z]+_3d|halo_put_dev)[^ ]*(_l[0-9]+):.*REG:([0-9]+).*/\1\2 REG=\3/'
for d in l_halo l_new; do g=$(ls -t $d/*.s | head -1); echo "== $d memory ops in device image:"; grep -oE "(st|ld|atom|fence|red)\.[a-z]+\.(sys|gpu)[a-z0-9.]*|fence\.sc\.sys|nanosleep[a-z0-9.]*|globaltimer|trap" $g | sort | uniq -c | tr '\n' ';'; echo; done
