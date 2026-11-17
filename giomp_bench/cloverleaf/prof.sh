#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
C=/shared/data1/Users/l1065028/giomp_bench/cloverleaf
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G CLOVER_GIOMP_PROF=1
cd $C/runs
for d in clover_bm_short.in clover_bm16_short.in; do
  srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 $C/giomp_prof --file $C/src/InputDecks/$d --out p.$d.out --staging-buffer false --profile 2>&1 </dev/null | grep -E "GIOMP PROF|Wall clock"
done
