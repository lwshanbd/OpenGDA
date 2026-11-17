#!/bin/bash
# MiniMod 1000^3, 1000 steps, 3 nodes: global barrier vs neighbour signals.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 8 -N3"
M=/shared/data1/Users/l1065028/giomp_bench/mm
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus 3"
X=/shared/data1/Users/l1065028/giomp_bench/final4/sig/mm_sig
for r in 1 2 3; do
  echo "MM mpi r$r: $($S $M/mm_mpi $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | tr '\n' ' ')"
  for m in 0 5 4; do
    echo "MM mode$m r$r: $(GICC_HALO_OVERLAP=$m $S $X $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
  done
done
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
for r in 1 2 3; do for b in jm jl5 jl_sig; do
  echo "JC $b r$r: $($S $J/$b -niter 1000 2>&1 | grep -E 'GPUs:|phases|MISMATCH|ERROR|rror' | tr '\n' ' ')"
done; done
echo SIG_DONE
