#!/bin/bash
# run_env.sh N CLASS VARIANT "ENV=VAL ..." TAG
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/npbft; mkdir -p out
env $4 srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N$1 -n$1 ./$3 -class $2 > out/n$1.$2.$3.$5.out 2>&1 </dev/null
echo "N=$1 $2 $3 [$4]: $(grep -h 'Verification\|Time in sec\|iterations (max\|rror' out/n$1.$2.$3.$5.out | tr -s ' ' | tr '\n' ' ')" >> run.log
