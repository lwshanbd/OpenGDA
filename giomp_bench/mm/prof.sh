#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/mm
NSYS=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/nsys
S="srun -p maple --account=app -N1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
$S $NSYS profile -t cuda -f true -o q_mpi ./mm_mpi --grid 1000 --nsteps 200 --niters 1 --ngpus 1 >/dev/null 2>&1
GICC_HALO_OVERLAP=0 $S $NSYS profile -t cuda -f true -o q_m0 ./mm_plain_lto --grid 1000 --nsteps 200 --niters 1 --ngpus 1 >/dev/null 2>&1
for f in q_mpi q_m0; do $NSYS export -t sqlite -f true -o $f.sqlite $f.nsys-rep >/dev/null 2>&1; done
