#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP_PROF=1 HYPRE_GIOMP=1
cat /proc/sys/kernel/perf_event_paranoid
srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 bash -c "perf record -F 4000 -g -o $H/perf/giomp.\$SLURM_PROCID.data $H/inst-giomp.struct -n 256 256 256 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1" 2>&1 | grep -E "wall|FIRST|perf|rror"
export HYPRE_GIOMP=0
srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2 bash -c "perf record -F 4000 -g -o $H/perf/base.\$SLURM_PROCID.data $H/inst-base.struct -n 256 256 256 -P 2 1 1 -solver 1 -rap 2 -gpu_mpi 1" 2>&1 | grep -E "wall|perf|rror"
