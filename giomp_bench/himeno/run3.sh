#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY HIMENO_PER_RANK=1
cd /shared/data1/Users/l1065028/giomp_bench/himeno
for k in 0 1 2 3; do echo "== skip=$k"; HIMENO_SKIP=$k srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N3 -n3 ./hl -size XL -niter 1000 2>&1 | grep -h "rank "; done
