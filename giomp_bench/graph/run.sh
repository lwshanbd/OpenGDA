#!/bin/bash
# run.sh N "ARGS_BFS" "ARGS_PR" REPS TAG: each variant REPS times on N nodes,
# interleaved, as srun steps (inside one salloc, or alone).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/graph; mkdir -p out
n=$1; ab=$2; ap=$3; reps=${4:-1}; tag=${5:-x}
R="srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$n -n$n"
for i in $(seq 1 $reps); do
  for v in bfs_mpi bfs_giomp; do
    [ -n "$ab" ] || continue
    $R ./$v $ab > out/$tag.n$n.$v.$i.out 2>&1 </dev/null
    echo "N=$n $v run$i ($ab): $(grep -h 'mean\|checksum\|rror' out/$tag.n$n.$v.$i.out | tr '\n' ' ')" >> run.log
  done
  for v in pagerank_mpi pagerank_giomp; do
    [ -n "$ap" ] || continue
    $R ./$v $ap > out/$tag.n$n.$v.$i.out 2>&1 </dev/null
    echo "N=$n $v run$i ($ap): $(grep -h 'median\|checksum\|rror' out/$tag.n$n.$v.$i.out | tr '\n' ' ')" >> run.log
  done
done
echo "RUN_DONE $tag" >> run.log
