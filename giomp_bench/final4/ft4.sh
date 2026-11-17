#!/bin/bash
# FT class C, 4 nodes: MPI, GiOMP, GiOMP + pipeline, LLM-tuned; all in one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/final4
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N4"
F=/shared/data1/Users/l1065028/giomp_bench/npbft
for r in 1 2 3; do
  for v in "mpi_isend $F/fm1" "mpi_alltoall $F/fm2" "giomp $F/fl" "giomp_pipeline /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/ft/rounds/5/b0/exe" "llm_tuned /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/ft/rounds/5/b1/exe"; do set -- $v
    echo "FT $1 r$r: $($S $2 -class C 2>&1 | grep -E 'Time in seconds|Verification|rror' | tr -s ' ' | tr '\n' ' ')"
  done
done
echo FT4_DONE
