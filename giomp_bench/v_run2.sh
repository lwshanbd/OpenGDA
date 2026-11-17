#!/bin/bash
# v_run2.sh: minimod mode 2, old vs versioned, alternated three times.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
cd /shared/data1/Users/l1065028/giomp_bench/mm
for i in 1 2 3; do for b in mm_pipe mm_pipe_v; do
  echo "MM $b m2 #$i: $(GICC_HALO_OVERLAP=2 $S -N1 ./$b --grid 1000 --nsteps 1000 --niters 3 --ngpus 1 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//; s/comm=.*//' | sort -u | tr '\n' ' ')"
done; done
