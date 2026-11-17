#!/bin/bash
# b1.sh: pipelined Jacobi with the uncalled helpers dropped (default
# register cap and mirror test), with save-temps, plus a one-GPU run.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
CO=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump
cd $J/fix
export GICC_DIST_CHUNK=8192
GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/pipe_drop | grep -E "FAIL|BUILT"
$CO -res-usage $PWD/*.cubin 2>/dev/null | grep -A1 "Function __omp" | paste - - | grep sweep | sed -E 's/.*REG:([0-9]+).*/REG=\1/'
grep -c "bar.sync" *.s
cd $J/..
bash run.sh 1 $J/fix/pipe_drop | grep -E "1 GPU:|rror"
