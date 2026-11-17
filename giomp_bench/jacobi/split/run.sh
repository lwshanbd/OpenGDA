#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/jacobi/split
L="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2"
for r in 0 1; do
  for nc in 1 100000; do
    for b in jn js; do
      echo "== $b nccheck=$nc r$r"; env OMPX_SPLIT_STATS=1 $L ./$b -niter 1000 -nccheck $nc -noref 2>&1 | grep -E "GPUs:|split launches|fence|FAIL|rror" | head -6
    done
  done
done
