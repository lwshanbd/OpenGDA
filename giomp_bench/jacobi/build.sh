#!/bin/bash
# build.sh: the OpenMP+MPI baseline and the GiOMP port plain and pipelined,
# all with the same kernels and -foffload-lto, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
GIOMP_EXTRA_FLAGS="-foffload-lto" PLAIN=1 bash $S $PWD/jacobi_mpi_omp.cpp $PWD/jm | tail -1
GIOMP_EXTRA_FLAGS="-foffload-lto" PLAIN=1 bash $S $PWD/jacobi_giomp.cpp $PWD/jl | tail -1
bash $S $PWD/jacobi_giomp.cpp $PWD/jp | grep -E "gicc-after|ILLEGAL|stays|rror|BUILT" | sort | uniq
