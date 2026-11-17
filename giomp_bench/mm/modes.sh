#!/bin/bash
# modes.sh: the GiOMP plain build with the original's separate kernels,
# blocking (mode 0) and with the interior overlapped by hand (mode 1).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
for N in 1 2 3; do for m in 0 1; do
  [ $N = 1 ] && [ $m = 1 ] && continue
  t=$(GICC_HALO_OVERLAP=$m $S -N$N ./mm_plain_lto --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//' | sort -u | tr '\n' ' ')
  echo "N=$N mm_plain_lto m$m: $t"
done; done
