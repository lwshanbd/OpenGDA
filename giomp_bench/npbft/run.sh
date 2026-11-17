#!/bin/bash
# run.sh N "CLASSES" "VARIANTS": every class with every variant on N nodes.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/npbft; mkdir -p out
n=$1
for c in $2; do for v in $3; do
  srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N$n -n$n ./$v -class $c > out/n$n.$c.$v.out 2>&1 </dev/null
  echo "N=$n $c $v: $(grep -h 'Verification\|Time in sec\|iterations (max\|rror' out/n$n.$c.$v.out | tr -s ' ' | tr '\n' ' ')" >> run.log
done; done
echo "RUN_DONE $n" >> run.log
