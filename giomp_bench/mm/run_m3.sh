#!/bin/bash
# run_m3.sh: minimod at GICC_HALO_OVERLAP=3 against the baseline and mode 0/1.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
cd /shared/data1/Users/l1065028/giomp_bench/mm
for N in 2 1; do
  echo "MM N=$N mm_mpi: $($S -N$N ./mm_mpi --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E 'rank 0 Time (comm|comp|modeling)|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
  for v in "mm_plain3 0" "mm_plain3 1" "mm_plain3 3" "mm_ctrl3 3" "mm_pipe3 3"; do set -- $v
    [ $N = 1 ] && [ $2 = 1 ] && continue
    echo "MM N=$N $1 m$2: $(GICC_HALO_OVERLAP=$2 $S -N$N ./$1 --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//' | sort -u | tr '\n' ' ')"
  done
done
