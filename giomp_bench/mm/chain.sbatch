#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 25
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm
for r in 1 2; do
  echo "CH N=$N chain r$r: $(OMPX_SPLIT_CHAIN_DEBUG=1 OMPX_SPLIT_STATS=1 GICC_HALO_OVERLAP=3 $S ./mm_chain $A 2>&1 | grep -E "^ITER|split launches|chain|dim [0-9]|field_xor|rror" | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "CH N=$N nochain r$r: $(OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STATS=1 GICC_HALO_OVERLAP=3 $S ./mm_chain $A 2>&1 | grep -E '^ITER|split launches|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "CH N=$N mode1 r$r: $(GICC_HALO_OVERLAP=1 $S ./mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
done
echo CH_DONE
