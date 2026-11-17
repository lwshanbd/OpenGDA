#!/bin/bash
# build.sh: MoE layer, MPI 1/2 and GiOMP 0/1/2, same kernels, -foffload-lto,
# no pass plugin, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
T=/shared/data1/Users/l1065028/giomp_ib_test
cd /shared/data1/Users/l1065028/giomp_bench/moe
export LIBDIR=$T/lib5
for v in ${VARIANTS:-mpi1 mpi2 mpi3 giomp0 giomp1 giomp2 giomp3}; do
  case $v in mpi*) D="-DMOE_MPI=${v#mpi}";; giomp*) D="-DMOE_GIOMP=${v#giomp}";; esac
  GIOMP_EXTRA_FLAGS="-foffload-lto $D ${EXTRA:-}" PLAIN=1 bash $S $PWD/moe.cpp $PWD/moe_$v${SUFFIX:-} | tail -5 &
done; wait
