#!/bin/bash
# run.sh N REPS TAG ARGS...: every variant REPS times on N nodes, interleaved.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=${OMPX_HEAP_SIZE:-4G}
cd /shared/data1/Users/l1065028/giomp_bench/moe; mkdir -p out
n=$1; reps=$2; tag=$3; shift 3
R="srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N$n -n$n"
for i in $(seq 1 $reps); do
  for v in ${VARIANTS:-mpi1 mpi2 mpi3 giomp0 giomp1 giomp2 giomp3}; do
    $R ./moe_$v "$@" > out/$tag.n$n.$v.$i.out 2>&1 </dev/null
    echo "$tag N=$n $v run$i: $(grep -h 'median\|phases\|rror\|rap\|abort' out/$tag.n$n.$v.$i.out | head -3 | tr '\n' ' ')" >> run.log
  done
done
echo "RUN_DONE $tag" >> run.log
