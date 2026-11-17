#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 50
# MiniMod 1000^3, 1000 steps, barrier before the timer in every build: the
# plain code with split + chain against the two hand-written overlaps, 5
# interleaved runs each, in one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm/barrier
for r in 1 2 3 4 5; do
  echo "RP N=$N giomp_plain_split_chain_stable r$r: $(GICC_HALO_OVERLAP=5 OMPX_SPLIT_STABLE=1 $S ./minimod_giomp_split_barrier $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "RP N=$N giomp_plain_split_chain r$r: $(GICC_HALO_OVERLAP=5 $S ./minimod_giomp_split_barrier $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "RP N=$N giomp_hand_overlap r$r: $(GICC_HALO_OVERLAP=1 $S ./minimod_giomp_nopass_barrier $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "RP N=$N mpi_hand_overlap r$r: $($S ./minimod_mpi_overlap_progress_barrier $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
  : echo "RP N=$N mpi_plain r$r: $($S ./minimod_mpi_plain_barrier $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
done
echo RP_DONE
