#!/bin/bash
# MiniMod 350^3, 2000 steps, 3 nodes: MPI, GiOMP, GiOMP + pipeline, LLM-tuned; one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/final4
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N3"
M=/shared/data1/Users/l1065028/giomp_bench/mm
A="--grid 350 --nsteps 2000 --niters 1 --ngpus 3"
G="OMP_NUM_TEAMS=1056 OMP_TEAMS_THREAD_LIMIT=256"
for r in 1 2 3; do
  echo "MM mpi r$r: $($S $M/mm_mpi $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | tr '\n' ' ')"
  echo "MM mpi_dc r$r: $($S $M/mm_mpi_dc $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | tr '\n' ' ')"
  echo "MM mpi_geom r$r: $(env $G $S $M/mm_mpi $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | tr '\n' ' ')"
  echo "MM giomp r$r: $(GICC_HALO_OVERLAP=2 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/9/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
  echo "MM giomp_pipeline r$r: $(GICC_HALO_OVERLAP=2 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/10/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
  echo "MM llm_tuned r$r: $(env $G GICC_HALO_OVERLAP=2 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/10/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | tr '\n' ' ')"
done
echo MM_DONE
