#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH OMPX_HEAP_SIZE=2G HYPRE_GIOMP=1 HYPRE_GIOMP_PROF=1 OMP_TARGET_OFFLOAD=MANDATORY
H=/shared/data1/Users/l1065028/giomp_bench/hypre
srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N${N:-2} $H/inst-giomp.struct -n 256 256 256 -P ${P:-2 1 1} -solver 1 -rap 2 2>&1 | grep -E "PROF|wall clock|Iterations"
