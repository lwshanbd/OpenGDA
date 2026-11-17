#!/bin/bash
# run_ov.sh N: MPI MiniMod (original, overlapped, overlapped + progress thread)
# against GiOMP mode 0 / 1, 1000^3, 1000 steps, 3 runs each, one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$1
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm
for r in 1 2 3; do
  for b in mm_mpi mm_mpi_new mm_mpi_ov1 mm_mpi_ov2; do
    echo "OV N=$N $b r$r: $($S ./$b $A 2>&1 | grep -E 'rank 0 Time (comm|modeling)|field_xor|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
  done
  for m in 0 1; do
    echo "OV N=$N giomp_m$m r$r: $(GICC_HALO_OVERLAP=$m $S ./mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  done
done
echo OV_DONE
