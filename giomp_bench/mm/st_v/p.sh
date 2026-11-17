#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_v
NSYS=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/nsys
S="srun -p maple --account=app -N1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
for v in ver nover; do
  GICC_HALO_OVERLAP=2 $S $NSYS profile -t cuda -f true -o q_$v ./$v/mm --grid 1000 --nsteps 200 --niters 1 --ngpus 1 2>&1 | grep -E 'field|modeling' | head -2
  $NSYS stats -r cuda_gpu_kern_sum -f csv q_$v.nsys-rep 2>/dev/null | awk -F'","' 'NR>1{gsub(/"/,"");n=$9; sub(/__omp_offloading_[0-9a-f_]+_/,"",n); printf "%s %s total=%s n=%s avg=%s\n","'$v'",substr(n,1,40),$2,$3,$4}'
done
