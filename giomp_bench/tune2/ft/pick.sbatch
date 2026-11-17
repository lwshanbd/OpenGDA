#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 10
# Which MPI layout of FT is the baseline: both at default choices, 3 runs.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 3 -N2"
cd /shared/data1/Users/l1065028/giomp_bench/tune2/ft
for r in 1 2 3; do for b in ft_mpi_plain ft_mpi_stream; do
  echo "PK $b r$r: $($S ./$b -class C 2>&1 | grep -E 'Time in seconds|Verification' | tr -s ' ' | tr '\n' ' ')"
done; done
echo PK_DONE
