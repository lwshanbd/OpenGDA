#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 20
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
for r in 1 2; do
  echo "SS N=$N split r$r: $(OMPX_SPLIT_STATS=1 GICC_HALO_OVERLAP=3 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_1000_m3/rounds/5/b1/exe $A 2>&1 | grep -E '^ITER|split launches|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "SS N=$N default r$r: $(GICC_HALO_OVERLAP=3 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_1000_m3/rounds/5/b0/exe $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "SS N=$N mode1 r$r: $(GICC_HALO_OVERLAP=1 $S ./mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
done
echo SS_DONE
