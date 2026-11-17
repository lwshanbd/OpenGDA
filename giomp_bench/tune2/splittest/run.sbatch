#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 20
# Correctness of the per-kernel split: modes 3 (a pipelined kernel sends every
# step) and 5 (host puts), split on and off, field_xor must match.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N2"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus 2"
cd /shared/data1/Users/l1065028/giomp_bench/tune2/splittest
for m in 3 5; do for sp in 1 0; do
  echo "ST mode=$m split=$sp: $(GICC_HALO_OVERLAP=$m OMPX_SPLIT=$sp OMPX_SPLIT_STATS=1 $S ./mm_split_default $A 2>&1 | grep -E '^ITER|rank 0: [0-9]+ split|rank 0: chain|rank 0: [0-9]+ deferred|field_xor|rror|Abort' | sed 's/straggler.*//; s/last window.*//' | tr '\n' ' ')"
done; done
echo ST_DONE
