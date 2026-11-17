#!/bin/bash
cd /shared/data1/Users/l1065028/giomp_bench
for N in 2 3; do for r in 1 2; do for b in ds/jm_w ds/jl_w ds/jp_w ds/jm_c8192; do
  echo "N=$N $b: $(bash run.sh $N jacobi/$b 2>&1 | grep -E 'GPUs:|phases|MISMATCH|rror' | tr '\n' ' ')"
done; done; done
