#!/bin/bash
# run_split.sh "VARIANTS" [ENV...]: Himeno XL on 3 nodes, each variant once.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/himeno; mkdir -p out
for v in $1; do
  env ${@:2} srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N3 -n3 ./$v -size XL -niter 1000 > out/split.$v.out 2>&1 </dev/null
  echo "$v [${@:2}]: $(grep -h 'cpu :\|Gosa\|phases\|field_xor\|rror\|giomp\]' out/split.$v.out | tr -s ' ' | tr '\n' ' ')" >> split.log
done
