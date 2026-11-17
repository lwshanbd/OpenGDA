#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 45
# MiniMod 1000^3, 1000 steps, 3 nodes, every build's inner kernel capped at
# 72 registers: plain MPI, MPI hand overlap, GiOMP plain (mode 5, no pass),
# GiOMP hand overlap (mode 1), GiOMP pipeline (mode 5, split + chain +
# STABLE, fence on neighbours); 4 interleaved runs each.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm/regs72
G='^ITER|field_xor|rror'
M='rank 0 Time modeling|field_xor|rror'
for r in 1 2 3 4; do
  echo "RG N=$N mpi_plain r$r: $($S ./minimod_mpi_plain $A 2>&1 | grep -E "$M" | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
  echo "RG N=$N mpi_hand_overlap r$r: $($S ./minimod_mpi_overlap_progress $A 2>&1 | grep -E "$M" | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
  echo "RG N=$N giomp_plain r$r: $(GICC_HALO_OVERLAP=5 $S ./minimod_giomp_nopass $A 2>&1 | grep -E "$G" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "RG N=$N giomp_hand_overlap r$r: $(GICC_HALO_OVERLAP=1 $S ./minimod_giomp_nopass $A 2>&1 | grep -E "$G" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "RG N=$N giomp_pipeline r$r: $(GICC_HALO_OVERLAP=5 OMPX_SPLIT_STABLE=1 $S ./minimod_giomp_split $A 2>&1 | grep -E "$G" | sed 's/straggler.*//' | tr '\n' ' ')"
done
echo RG_DONE
