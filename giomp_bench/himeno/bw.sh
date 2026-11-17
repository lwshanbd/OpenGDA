#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/himeno
for sd in "0 0" "4 0" "0 4" "4 4" "32 32" "64 64" "8 8" "16 16"; do
  set -- $sd
  echo "== src+$1 dst+$2"; srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N3 -n3 ./putbw 2101248 300 0 0 0 $1 $2 2>&1 | grep "rank 1"
done
