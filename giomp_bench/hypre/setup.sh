#!/bin/bash
# setup.sh: where does GiOMP's extra setup time go (256^3 PFMG, 2 nodes).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2"
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP_PROF=1
for rep in 1 2; do
for v in "base_gpu:base:1:0:" "giomp:giomp:1:1:" "giomp_small:giomp:1:1:OMPX_HEAP_SIZE=256M HYPRE_GIOMP_ARENA=128M HYPRE_GIOMP_SEND=32M" "giomp_nomtc:giomp:1:1:UCX_MEMTYPE_CACHE=n" "base_nomtc:base:1:0:UCX_MEMTYPE_CACHE=n"; do
  IFS=: read name tree gm g extra <<< "$v"
  out=$(env HYPRE_GIOMP=$g $extra $R $H/inst-$tree.struct -n 256 256 256 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi $gm 2>&1 </dev/null)
  echo "== $name rep=$rep"; echo "$out" | grep -E "wall clock|FIRST|Iterations"
done; done
