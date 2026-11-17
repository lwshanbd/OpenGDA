#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP_PROF=1 HYPRE_GIOMP=1
for n in 256 384; do for s in 1 11; do
echo "== n=$n solver=$s"
srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 $H/inst-giomp.struct -n $n $n $n -P 2 1 1 -solver $s -rap 2 -gpu_mpi 1 2>&1 | grep -E "wall clock|PACK|COMM rank|PROF"
done; done
