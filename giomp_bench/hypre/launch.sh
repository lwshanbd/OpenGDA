#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP_PROF=1 HYPRE_GIOMP=1
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2"
A="-n 256 256 256 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1"
for rep in 1 2; do
echo "== direct"; $R $H/inst-giomp.struct $A 2>&1 | grep -E "wall clock|FIRST" | head -4
echo "== bash"; $R bash -c "$H/inst-giomp.struct $A" 2>&1 | grep -E "wall clock|FIRST" | head -4
echo "== perf"; $R bash -c "perf record -F 4000 -g -o /tmp/p.data $H/inst-giomp.struct $A" 2>&1 | grep -E "wall clock|FIRST" | head -4
echo "== perf_stat"; $R bash -c "perf stat -o /dev/null $H/inst-giomp.struct $A" 2>&1 | grep -E "wall clock|FIRST" | head -4
done
