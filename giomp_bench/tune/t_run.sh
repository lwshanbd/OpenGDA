#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
cd /shared/data1/Users/l1065028/giomp_bench/tune
export OMP_TARGET_OFFLOAD=MANDATORY
for x in mw_t mw_t2; do
  srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 ./$x 2>&1 </dev/null | grep -E 'CPU Time|d_mass|d_te|rror'
done
LIBOMPTARGET_PROFILE=$PWD/prof_%r.json srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 bash -c 'LIBOMPTARGET_PROFILE=$PWD/prof_$SLURM_PROCID.json ./mw_t' 2>&1 </dev/null | grep -E 'CPU Time|rror'
ls -la prof_*.json 2>&1 | head
