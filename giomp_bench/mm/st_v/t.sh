#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_v
S="srun -p maple --account=app -N1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
for i in 1 2; do for v in ${VARS:-nover ver noinl}; do
  echo "$v m${M:-2} #$i: $(GICC_HALO_OVERLAP=${M:-2} $S ./$v/mm --grid 1000 --nsteps 400 --niters 1 --ngpus 1 2>&1 | grep -E '^ITER 0|rror|field' | sed 's/comm=.*//' | sort -u | tr '\n' ' ')"
done; done
