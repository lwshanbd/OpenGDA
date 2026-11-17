#!/bin/bash
# b.sh: the three Jacobi builds with every kernel's distribute chunk widened.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
P=/shared/data1/Users/l1065028/giomp_ib_test/passes-build/libgicc-passes.so
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
export GICC_DIST_CHUNK=8192
GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P" PLAIN=1 bash $S $PWD/jacobi_mpi_omp.cpp $PWD/ds/jm_w 2>&1 | grep -E "dist-chunk|BUILT|FAIL|rror" | sort | uniq
GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P" PLAIN=1 bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jl_w 2>&1 | grep -E "dist-chunk|BUILT|FAIL|rror" | sort | uniq
bash $S $PWD/jacobi_giomp.cpp $PWD/ds/jp_w 2>&1 | grep -E "dist-chunk|BUILT|FAIL|rror" | sort | uniq
