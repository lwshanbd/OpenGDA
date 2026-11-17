#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
make -C /shared/data1/Users/l1065028/giomp_ib_test/passes-build -j16 2>&1 | grep -E "error|Built target"
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
CO=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump
export GICC_DIST_CHUNK=8192
cd $J/st3/io && GICC_X_IDLEONLY=1 GICC_X_CAP=64 GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL
$CO -res-usage $J/st3/io/*.cubin | grep -A1 "Function __omp.*sweep" | grep -oE "REG:[0-9]+" 
cd $J/..; for b in jacobi/ds/jl_w jacobi/st3/io/x jacobi/st3/pw/x; do echo "$b: $(bash run.sh 2 $b | grep -E 'GPUs:' | sed -E 's/.*1 GPU: *([0-9.]+) s, *[0-9]+ GPUs: *([0-9.]+) s.*/1gpu=\1 2gpu=\2/')"; done
