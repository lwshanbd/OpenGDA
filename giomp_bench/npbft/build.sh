#!/bin/bash
# build.sh: the OpenMP target + MPI baselines and the GiOMP port, all with
# the same kernels and -foffload-lto, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
T=/shared/data1/Users/l1065028/giomp_ib_test
cd /shared/data1/Users/l1065028/giomp_bench/npbft
export LIBDIR=$T/lib5
[ -n "$ONLY_STREAM" ] || GIOMP_EXTRA_FLAGS="-foffload-lto -DFT_MPI=1" PLAIN=1 bash $S $PWD/ft_giomp.cpp $PWD/fm1 | tail -3
[ -n "$ONLY_STREAM" ] || GIOMP_EXTRA_FLAGS="-foffload-lto -DFT_MPI=2" PLAIN=1 bash $S $PWD/ft_giomp.cpp $PWD/fm2 | tail -3
[ -n "$ONLY_STREAM" ] || GIOMP_EXTRA_FLAGS="-foffload-lto" PLAIN=1 bash $S $PWD/ft_giomp.cpp $PWD/fl | tail -3
GIOMP_EXTRA_FLAGS="-foffload-lto -DFT_MPI=1 -DFT_STREAM=1" PLAIN=1 bash $S $PWD/ft_giomp.cpp $PWD/fms | tail -3
GIOMP_EXTRA_FLAGS="-foffload-lto -DFT_STREAM=1" PLAIN=1 bash $S $PWD/ft_giomp.cpp $PWD/fls | tail -3
GIOMP_EXTRA_FLAGS="-DFT_STREAM=1" bash $S $PWD/ft_giomp.cpp $PWD/fps | sort | uniq | tail -20
