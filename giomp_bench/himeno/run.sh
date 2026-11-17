#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/himeno; mkdir -p out
for n in 1 2 3; do for s in L XL; do for v in hm hl hp; do
  [ $n = 1 ] && [ $v != hm ] && [ $v != hl ] && continue
  srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N$n -n$n ./$v -size $s -niter 1000 > out/n$n.$s.$v.out 2>&1 </dev/null
  echo "N=$n $s $v: $(grep -h 'cpu :\|phases\|field_xor\|rror' out/n$n.$s.$v.out | tr '\n' ' ')" >> run.log
done; done; done
echo RUN_DONE >> run.log
