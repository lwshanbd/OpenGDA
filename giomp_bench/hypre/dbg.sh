#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP=1 HYPRE_GIOMP_PROF=1 HYPRE_GIOMP_PIPE=1
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N2"
$R $H/pipe18/struct -n 64 64 64 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1 2>&1 | grep -vE "^\[maple.*\] \[ *[0-9]+\]" | tail -40
echo "== check mode 1"
HYPRE_GIOMP_CHECK=1 $R $H/pipe18/struct -n 64 64 64 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1 2>&1 | grep -vE "^\[maple.*\] \[ *[0-9]+\]" | grep -v "^ *$" | head -30
