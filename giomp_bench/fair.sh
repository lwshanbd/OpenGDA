#!/bin/bash
# fair.sh: every benchmark against its OpenMP+MPI baseline, all builds with
# the same kernels and -foffload-lto.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
B=/shared/data1/Users/l1065028/giomp_bench
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
cd $B/mm
for N in 1 2 3; do
  echo "MM N=$N mm_mpi: $($S -N$N ./mm_mpi --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E 'rank 0 Time (comm|comp|modeling)|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ') xor=$($S -N$N ./mm_mpi --grid 1000 --nsteps 1000 --niters 1 --ngpus $N 2>&1 | grep -m1 field_xor | awk '{print $NF}')"
  for v in "mm_plain_lto 0" "mm_plain_lto 1" "mm_plain_lto 2" "mm_ctrl 2" "mm_pipe 2"; do set -- $v
    [ $N = 1 ] && [ $2 = 1 ] && continue
    echo "MM N=$N $1 m$2: $(GICC_HALO_OVERLAP=$2 $S -N$N ./$1 --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//' | sort -u | tr '\n' ' ')"
  done
done
cd $B
for N in 2 3; do for r in 1 2; do for b in jm jl jp; do
  echo "JAC N=$N $b: $(bash run.sh $N jacobi/$b 2>&1 | grep -E 'GPUs:|phases|MISMATCH|ERROR|rror' | tr '\n' ' ')"
done; done; done
cd $B/miniweather
for N in 2 3; do for b in mw_mpi mw_plain mw_pipe; do
  echo "MW N=$N $b: $($S -N$N ./$b 2>&1 | grep -E 'CPU Time|d_mass|d_te|rror' | tr '\n' ' ')"
done; done
