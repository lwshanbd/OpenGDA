#!/bin/bash
# r1.sh: every Jacobi variant on one GPU (no peer mapped), twice.
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
cd $J/..
for r in 1 2; do
for v in ds/jm_w ds/jl_w fix/pipe_drop fix/ver fix/cap72 fix/cap72_nomirror fix/ver_cap72 fix/cap72_nomirror_nograb fix/cap72_nomirror_nocount; do
  echo "$v: $(bash run.sh 1 $J/$v | grep -E '1 GPU:|rror' | sed -E 's/.*1 GPU: *([0-9.]+) s.*/\1/' | tr '\n' ' ')"
done; done
