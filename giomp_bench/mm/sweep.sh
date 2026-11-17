#!/bin/bash
# sweep.sh: MiniMod grid 1000 on 1-3 nodes, the OpenMP+MPI baseline and the
# three GiOMP builds (plain, rewritten kernel with every put deferred, pipelined).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
for N in 1 2 3; do for b in ./mm_mpi ./mm_plain_lto ./mm_ctrl /shared/data1/Users/l1065028/giomp_ib_test/mm/mm_pipe; do
  t=$(GICC_HALO_OVERLAP=2 $S -N$N $b --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E '^ITER 2|rank 0 Time|rror|field' | sed 's/straggler.*//' | sort -u | tr '\n' ' ')
  echo "N=$N $(basename $b): $t"
done; done
