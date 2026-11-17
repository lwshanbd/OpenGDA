#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 15
# Step 6, kernel level: GPU time of fftz_send in the MPI streamed build with
# and without C_giomp's teams/chunk (communication noise does not reach it).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
N=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/compilers/bin/nsys
cd /shared/data1/Users/l1065028/giomp_bench/tune2/ft
for b in xval_mpi_stream_default xval_mpi_stream_cgiomp; do
  srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 4 -N2 $N profile -t cuda -o $b.%q{SLURM_PROCID} --force-overwrite true ./$b -class C > $b.prof.out 2>&1
  $N stats -q -r cuda_gpu_kern_sum -f csv -o $b --force-overwrite true $b.0.nsys-rep >/dev/null 2>&1
  echo "XN $b: $(grep -i fftz_send ${b}_cuda_gpu_kern_sum.csv | cut -d, -f2-4)"
done
echo XN_DONE
