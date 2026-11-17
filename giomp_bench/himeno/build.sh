#!/bin/bash
# build.sh: the OpenMP+MPI baseline and the GiOMP port plain and pipelined,
# all with the same kernels and -foffload-lto, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
cd /shared/data1/Users/l1065028/giomp_bench/himeno
GIOMP_EXTRA_FLAGS="-foffload-lto -DHIMENO_MPI=1" PLAIN=1 bash $S $PWD/himeno_giomp.cpp $PWD/hm | tail -3
GIOMP_EXTRA_FLAGS="-foffload-lto" PLAIN=1 bash $S $PWD/himeno_giomp.cpp $PWD/hl | tail -3
bash $S $PWD/himeno_giomp.cpp $PWD/hp | sort | uniq | tail -20
