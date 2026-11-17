#!/bin/bash
# b3.sh: pipelined Jacobi with the region versioned on the mirror test.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
cd $J/fix
export GICC_DIST_CHUNK=8192
bash $S $J/jacobi_giomp.cpp $PWD/ver 2>&1 | grep -E "FAIL|BUILT|unmirrored|rror"
GICC_X_CAP=72 bash $S $J/jacobi_giomp.cpp $PWD/ver_cap72 2>&1 | grep -E "FAIL|BUILT|rror"
