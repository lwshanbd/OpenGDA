#!/bin/bash
# run_base.sh N EXE ARGS...: N nodes, one GPU each.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
N=$1; X=$2; shift 2
srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10 -N$N $X "$@" 2>&1 </dev/null
