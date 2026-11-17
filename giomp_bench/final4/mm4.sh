#!/bin/bash
# MiniMod 1000^3, 1000 steps, 4 nodes. Every GiOMP build runs the seven region
# kernels of the MPI version (GICC_HALO_OVERLAP=3); one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 8 -N4"
M=/shared/data1/Users/l1065028/giomp_bench/mm
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus 4"
R=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_1000_m3/rounds
for r in 1 2 3; do
  echo "MM mpi r$r: $($S $M/mm_mpi $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | tr '\n' ' ')"
  echo "MM giomp r$r: $(GICC_HALO_OVERLAP=3 $S $R/1/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
  echo "MM giomp_pipeline r$r: $(GICC_HALO_OVERLAP=3 $S $R/2/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
  echo "MM llm_tuned r$r: $(GICC_HALO_OVERLAP=3 $S $R/5/b1/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
done
echo MM4_DONE
