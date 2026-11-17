#!/bin/bash
# prof.sh: rank 0 of a 2-node run of each Jacobi build under nsys.
cd /shared/data1/Users/l1065028/giomp_bench/jacobi
for b in jm jl jp; do bash nsys.sh 2 q_$b ./$b -niter 300 -noref >/dev/null 2>&1; done
