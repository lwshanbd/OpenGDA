#!/bin/bash
# regs.sh: register counts of each build's kernels, from the device link's save-temps.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
CO=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump
export GICC_DIST_CHUNK=8192
show() { echo "== $1"; $CO -res-usage $J/ds/st_$1/*.cubin 2>/dev/null | grep -A1 "Function __omp" | paste - - | grep -E "sweep" | sed -E 's/.*REG:([0-9]+).*/REG=\1/'; }
for v in jm_w jp_w jp_cap32m; do mkdir -p $J/ds/st_$v; done
cd $J/ds/st_jm_w && GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P -Wl,--device-linker=--save-temps" PLAIN=1 bash $S $J/jacobi_mpi_omp.cpp $PWD/x | grep FAIL; show jm_w
cd $J/ds/st_jp_w && GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL; show jp_w
cd $J/ds/st_jp_cap32m && GICC_X_CAP=32 GICC_X_NOMIRROR=1 GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL; show jp_cap32m
