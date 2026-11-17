#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 40
# MiniMod 1000^3, 1000 steps, 3 nodes: the plain code with split + chain +
# OMPX_SPLIT_STABLE, its deferred fence a barrier vs neighbours only, against
# the hand-written overlaps; 4 interleaved runs each.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
B=/shared/data1/Users/l1065028/giomp_bench/mm/barrier
cd /shared/data1/Users/l1065028/giomp_bench/mm/neighbors
F='^ITER|field_xor|rror|deferred fences'
for r in 1 2 3 4; do
  echo "NB N=$N pipeline_neighbours r$r: $(GICC_HALO_OVERLAP=5 OMPX_SPLIT_STABLE=1 OMPX_SPLIT_STATS=1 $S ./minimod_giomp_split $A 2>&1 | grep -E "$F" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "NB N=$N pipeline_barrier r$r: $(GICC_HALO_OVERLAP=5 OMPX_SPLIT_STABLE=1 OMPX_FENCE_NEIGHBORS=0 OMPX_SPLIT_STATS=1 $S ./minimod_giomp_split $A 2>&1 | grep -E "$F" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "NB N=$N giomp_hand_overlap r$r: $(GICC_HALO_OVERLAP=1 $S $B/minimod_giomp_nopass_barrier $A 2>&1 | grep -E "$F" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "NB N=$N mpi_hand_overlap r$r: $($S $B/minimod_mpi_overlap_progress_barrier $A 2>&1 | grep -E 'rank 0 Time modeling|field_xor|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
done
echo NB_DONE
