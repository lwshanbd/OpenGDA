#!/bin/bash
# ncu.sh BIN...: one sweep launch of each binary under Nsight Compute, one GPU.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
NCU=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/ncu
M=gpu__time_duration.sum,sm__inst_executed.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_op_global_ld.sum,dram__bytes.sum,l1tex__t_bytes.sum,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
for b in "$@"; do
  echo "== $b"
  srun -p maple --account=app -N 1 --gres=gpu:1 --cpus-per-task=4 --mpi=pmix -t 10 \
    $NCU -k regex:jacobi_sweep --launch-skip 20 --launch-count 1 --metrics $M $b -noref -niter 30 2>&1 | grep -E "^ +[a-z_]+__|==ERR|rror" 
done
