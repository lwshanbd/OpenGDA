#!/bin/bash
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
cd /shared/data1/Users/l1065028/giomp_ib_test/passes-build && make -j16 2>&1 | grep -E " error"
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
for c in 32 48 64; do GICC_X_CAP=$c GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P" PLAIN=1 bash $S $PWD/jacobi_mpi_omp.cpp $PWD/ds/jm_cap$c 2>&1 | grep FAIL; done
GICC_X_CAP=72 GICC_X_NOMIRROR=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_cap72m 2>&1 | grep -E "FAIL"
GICC_X_CAP=48 GICC_X_NOMIRROR=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_cap48m 2>&1 | grep -E "FAIL"
cd ..
for b in jm_cap32 jm_cap48 jm_cap64 jp_cap72m jp_cap48m; do echo "$b: $(bash run.sh 1 jacobi/ds/$b 2>&1 | grep -E '1 GPU:|rror' | sed 's/.*1 GPU: *//; s/,.*//' | tr '\n' ' ')"; done
