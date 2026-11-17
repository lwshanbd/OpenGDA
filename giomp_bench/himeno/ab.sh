#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/himeno
for n in 3 2; do for s in XL L; do for v in hm hl5 hp6 hp5; do
  echo "N=$n $s $v: $(srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N$n -n$n ./$v -size $s -niter 1000 2>&1 | grep -h 'cpu :\|phases\|field_xor' | tr '\n' ' ')"
done; done; done
