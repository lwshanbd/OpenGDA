#!/bin/bash
cd /shared/data1/Users/l1065028/giomp_bench
for c in 512 2048 8192 32768 131072; do
  echo "c=$c: $(bash run.sh 2 jacobi/ds/jm_c$c 2>&1 | grep -E 'GPUs:|MISMATCH|rror' | tr '\n' ' ')"
done
