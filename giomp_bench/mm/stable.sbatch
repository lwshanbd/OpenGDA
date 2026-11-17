#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 30
# MiniMod 1000^3, 1000 steps: the plain mode-3 code with split (the LLM's
# tuning file), with and without OMPX_SPLIT_STABLE / the chain, against the
# hand-written overlap (mode 1) and the MPI overlap with a progress thread.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm
B=./mm_chain
for r in 1 2; do
  for v in "split_stable OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STABLE=1" "split_stable_copy OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STABLE=1 OMPX_SPLIT_CTL_COPY=1" "split_stable_chain OMPX_SPLIT_STABLE=1" "split OMPX_SPLIT_CHAIN=0"; do
    set -- $v; name=$1; shift
    echo "ST N=$N $name r$r: $(env "$@" OMPX_SPLIT_STATS=1 GICC_HALO_OVERLAP=3 $S $B $A 2>&1 | grep -E '^ITER|split launches|chain:|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  done
  echo "ST N=$N giomp_mode1 r$r: $(GICC_HALO_OVERLAP=1 $S ./mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "ST N=$N mpi_overlap_progress r$r: $($S ./mm_mpi_ov2 $A 2>&1 | grep -E 'rank 0 Time (comm|modeling)|field_xor|rror' | tr '\n' ' ')"
done
echo ST_DONE
