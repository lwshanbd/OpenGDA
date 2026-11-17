#!/bin/bash
# teams.sh: sweep kernel time vs number of teams, baseline and pipelined (cap 72), 4096^2, one GPU.
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
NSYS=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/nsys
for t in 924 1848 3200; do for b in ds/jm_w fix/ver_cap72; do
  OMP_NUM_TEAMS=$t bash nsys.sh 1 fix/t_$t >/dev/null 2>&1 env OMP_NUM_TEAMS=$t $b -noref -nx 4096 -ny 4096
  g=$($NSYS stats -q --report cuda_gpu_trace --format csv fix/t_$t.nsys-rep 2>/dev/null | grep sweep | head -1 | cut -d, -f4)
  k=$($NSYS stats -q --report cuda_gpu_kern_sum --format csv fix/t_$t.nsys-rep 2>/dev/null | grep sweep | cut -d, -f4)
  echo "teams=$t grid=$g $b: sweep avg ${k} ns"
done; done
