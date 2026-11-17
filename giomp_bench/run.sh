#!/bin/bash
# run.sh N CMD...: N nodes, one rank and one GPU each, on Maple.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
N=$1; shift
srun -p maple --account=app -N $N --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mpi=pmix -t 10 "$@" 2>&1 | grep -vE "unknown CUDA"
