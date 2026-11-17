#!/bin/bash
# teams2.sh: total time vs number of teams, baseline / pipelined (cap 72, and default cap), one GPU.
cd /shared/data1/Users/l1065028/giomp_bench
for n in 4096 16384; do for t in 924 3200; do for b in ds/jm_w fix/ver_cap72 fix/ver; do
  echo "n=$n teams=$t $b: $(bash run.sh 1 env OMP_NUM_TEAMS=$t jacobi/$b -noref -nx $n -ny $n | grep -E 'GPUs: ' | sed -E 's/.*1 GPUs: *([0-9.]+) s.*/\1/')"
done; done; done
