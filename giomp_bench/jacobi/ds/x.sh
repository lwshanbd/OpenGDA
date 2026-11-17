#!/bin/bash
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
cd /shared/data1/Users/l1065028/giomp_ib_test/passes-build && make -j16 2>&1 | grep -E " error" 
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
GICC_X_NOMIRROR=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_xm 2>&1 | grep -E "BUILT|FAIL"
GICC_X_NOCAP=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_xc 2>&1 | grep -E "BUILT|FAIL"
GICC_X_NOCAP=1 GICC_X_NOMIRROR=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_xmc 2>&1 | grep -E "BUILT|FAIL"
cd ..
for b in jp_w jp_xm jp_xc jp_xmc jl_w; do echo "$b: $(bash run.sh 1 jacobi/ds/$b 2>&1 | grep -E '1 GPU:|rror' | sed 's/.*1 GPU: *//' | tr '\n' ' ')"; done
