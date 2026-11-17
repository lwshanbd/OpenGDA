#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
CO=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump
export GICC_DIST_CHUNK=8192
cd $J/st3/jl && GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P -Wl,--device-linker=--save-temps" PLAIN=1 bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL
cd $J/st3/pw && GICC_X_CAP=64 GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL
for v in jl pw; do echo == $v; $CO -res-usage $J/st3/$v/*.cubin | grep -A1 "Function __omp.*sweep" | grep -oE "REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | paste -sd' '; done
