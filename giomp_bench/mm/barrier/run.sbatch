#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 45
# MiniMod 1000^3, 1000 steps: each build with and without the barrier before
# the timer starts, interleaved, in one allocation.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm/barrier
giomp() {   # label binary env...
  local label=$1 bin=$2; shift 2
  echo "BR N=$N $label: $(env "$@" $S ./$bin $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
}
mpi() {   # label binary
  echo "BR N=$N $1: $($S ./$2 $A 2>&1 | grep -E 'rank 0 Time (comm|modeling)|field_xor|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
}
for r in 1 2; do
  for b in nobarrier barrier; do
    giomp "giomp_plain_$b r$r" minimod_giomp_nopass_$b GICC_HALO_OVERLAP=5
    giomp "giomp_plain_split_$b r$r" minimod_giomp_split_$b GICC_HALO_OVERLAP=5 OMPX_SPLIT_CHAIN=0
    giomp "giomp_plain_split_stable_$b r$r" minimod_giomp_split_$b GICC_HALO_OVERLAP=5 OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STABLE=1
    giomp "giomp_hand_overlap_$b r$r" minimod_giomp_nopass_$b GICC_HALO_OVERLAP=1
    mpi "mpi_plain_$b r$r" minimod_mpi_plain_$b
    mpi "mpi_hand_overlap_$b r$r" minimod_mpi_overlap_progress_$b
  done
done
echo BR_DONE
