#!/bin/bash
# prof.sh VARIANT: nsys kernel times of Himeno XL, 3 nodes, 100 iterations.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
NSYS=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/compilers/bin/nsys
cd /shared/data1/Users/l1065028/giomp_bench/himeno; mkdir -p prof; rm -f prof/$1.*
srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N3 -n3 \
  bash -c "$NSYS profile -t cuda,osrt -o prof/$1.\$SLURM_PROCID ./$1 -size XL -niter 100 > /dev/null 2>&1"
$NSYS stats --report cuda_gpu_kern_sum --format csv prof/$1.1.nsys-rep 2>/dev/null | cut -d, -f1-6,9 | sed 's/__omp_offloading_[0-9a-f_]*__ZN12_GLOBAL__N_1//' | head -12
$NSYS stats --report cuda_api_sum --format csv prof/$1.0.nsys-rep 2>/dev/null | cut -d, -f1-5,9 | head -14
