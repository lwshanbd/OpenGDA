#!/bin/bash
#SBATCH -p maple
#SBATCH -A app
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH -t 20
# Nsight Systems timelines of MiniMod, 200 steps, 2 ranks: the plain mode-3
# code with split and the stable-epoch lock, against the hand-written overlap.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
NSYS=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/compilers/bin/nsys
N=$SLURM_NNODES
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 6 -N$N"
A="--grid 1000 --nsteps 200 --niters 1 --ngpus $N"
cd /shared/data1/Users/l1065028/giomp_bench/mm/prof
OMPX_SPLIT_CHAIN=0 OMPX_SPLIT_STABLE=1 GICC_HALO_OVERLAP=3 $S $NSYS profile -t cuda,mpi -f true -o split_r%q{SLURM_PROCID} ../mm_chain $A 2>&1 | grep -E '^ITER|field_xor'
GICC_HALO_OVERLAP=1 $S $NSYS profile -t cuda,mpi -f true -o mode1_r%q{SLURM_PROCID} ../mm_plain_lto $A 2>&1 | grep -E '^ITER|field_xor'
for f in split_r0 split_r1 mode1_r0 mode1_r1; do
  $NSYS stats -q -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $f $f.nsys-rep >/dev/null 2>&1
done
echo PROF_DONE
