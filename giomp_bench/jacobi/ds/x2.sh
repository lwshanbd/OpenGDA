#!/bin/bash
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
cd /shared/data1/Users/l1065028/giomp_ib_test/passes-build && make -j16 2>&1 | grep -E " error"
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
for c in 32 40 48 64; do GICC_X_CAP=$c bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_cap$c 2>&1 | grep -E "FAIL"; done
GICC_X_CAP=32 GICC_X_NOMIRROR=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_cap32m 2>&1 | grep -E "FAIL"
cd ..
for b in jp_cap32 jp_cap40 jp_cap48 jp_cap64 jp_cap32m; do echo "$b: $(bash run.sh 1 jacobi/ds/$b 2>&1 | grep -E '1 GPU:|rror' | sed 's/.*1 GPU: *//; s/,.*//' | tr '\n' ' ')"; done
