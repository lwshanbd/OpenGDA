#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP=1 HYPRE_GIOMP_PROF=1
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 2 -N2"
for n in 64 128; do
for v in "giomp:$H/inst-giomp.struct:0" "lto_plain:$H/inst-pipe.struct:0" "lto_nopipe:$H/pipe18/struct:0" "pipe:$H/pipe18/struct:1"; do
  IFS=: read name exe pp <<< "$v"
  echo "== n=$n $name"
  HYPRE_GIOMP_PIPE=$pp $R $exe -n $n $n $n -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1 2>&1 | grep -E "UCX  ERROR|Iterations|Final Rel|GIOMP PIPE|GIOMP PACK" | head -6
done; done
