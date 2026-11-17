#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY GICC_HALO_OVERLAP=2
cd /shared/data1/Users/l1065028/giomp_bench/final4
A="--grid 350 --nsteps 2000 --niters 1 --ngpus 3"
S="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 2 -N3"
echo "== 1056x256"; OMP_NUM_TEAMS=1056 OMP_TEAMS_THREAD_LIMIT=256 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/9/b1/exe $A 2>&1 | grep -v "^ *$" | head -40
echo "== 1056x128"; OMP_NUM_TEAMS=1056 OMP_TEAMS_THREAD_LIMIT=128 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/9/b1/exe $A 2>&1 | grep -E "ITER|rror|stall|Abort" | head -5
echo "== default x256"; OMP_TEAMS_THREAD_LIMIT=256 $S /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_strong_untuned/rounds/9/b1/exe $A 2>&1 | grep -E "ITER|rror|stall|Abort" | head -5
