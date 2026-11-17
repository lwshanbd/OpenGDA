#!/bin/bash
# sweep.sh: the three Jacobi builds on 2 and 3 nodes, three runs each.
cd /shared/data1/Users/l1065028/giomp_bench
for N in 2 3; do for r in 1 2 3; do for b in jm jl jp; do
  echo "N=$N $b: $(bash run.sh $N jacobi/$b 2>&1 | grep -E 'GPUs:|phases|MISMATCH|ERROR|rror' | tr '\n' ' ')"
done; done; done
