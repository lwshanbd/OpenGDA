#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
cd $J/st3/t && GICC_X_CAP=64 GIOMP_EXTRA_FLAGS="-Wl,--device-linker=--save-temps" bash $S $J/jacobi_giomp.cpp $PWD/x | grep FAIL
