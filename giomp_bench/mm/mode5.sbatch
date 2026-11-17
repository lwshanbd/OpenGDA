#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 40
# MiniMod 1000^3, 1000 steps: the plainest code (mode 5: put the faces from
# the host, ompx_fence, the seven region kernels) built with split, against
# the hand-written overlaps.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N$N"
A="--grid 1000 --nsteps 1000 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm
for r in 1 2; do
  for v in "plain_nosplit OMPX_SPLIT=0" "plain_split OMPX_SPLIT_CHAIN=0" "plain_split_stable OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STABLE=1" "plain_split_chain OMPX_SPLIT_STABLE=1"; do
    set -- $v; name=$1; shift
    echo "M5 N=$N $name r$r: $(env "$@" OMPX_SPLIT_STATS=1 GICC_HALO_OVERLAP=5 $S ./mm_chain $A 2>&1 | grep -E '^ITER|split launches|chain:|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  done
  echo "M5 N=$N giomp_hand_overlap r$r: $(GICC_HALO_OVERLAP=1 $S ./mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor|rror' | sed 's/straggler.*//' | tr '\n' ' ')"
  echo "M5 N=$N mpi_hand_overlap r$r: $($S ./mm_mpi_ov2 $A 2>&1 | grep -E 'rank 0 Time (comm|modeling)|field_xor|rror' | tr '\n' ' ')"
  echo "M5 N=$N mpi_plain r$r: $($S ./mm_mpi $A 2>&1 | grep -E 'rank 0 Time (comm|modeling)|field_xor|rror' | tr '\n' ' ')"
done
echo M5_DONE
