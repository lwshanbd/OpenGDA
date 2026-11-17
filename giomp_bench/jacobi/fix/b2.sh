#!/bin/bash
# b2.sh: pipelined Jacobi ablations at the baseline's register count (72),
# then all of them on 2 nodes, twice.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
CO=/hrtc/apps/cuda/12.4.131/aarch64/rocky9/bin/cuobjdump
cd $J/fix
export GICC_DIST_CHUNK=8192
b() { local out=$1; shift; env "$@" bash $S $J/jacobi_giomp.cpp $PWD/$out | grep -E "FAIL"; }
b cap72               GICC_X_CAP=72
b cap72_nomirror      GICC_X_CAP=72 GICC_X_NOMIRROR=1
b cap72_nomirror_nograb  GICC_X_CAP=72 GICC_X_NOMIRROR=1 GICC_X_NOGRAB=1
b cap72_nomirror_nocount GICC_X_CAP=72 GICC_X_NOMIRROR=1 GICC_X_NOCOUNT=1
cd $J/..
for r in 1 2; do
for v in ds/jm_w ds/jl_w fix/pipe_drop fix/cap72 fix/cap72_nomirror fix/cap72_nomirror_nograb fix/cap72_nomirror_nocount; do
  echo "$v: $(bash run.sh 2 $J/$v | grep -E 'phases|2 GPUs:|rror' | sed -E 's/.*step ([0-9.]+) s, [a-z]+ ([0-9.]+) s.*/step \1 wait \2/; s/.*2 GPUs: *([0-9.]+) s.*/total \1/' | tr '\n' ' ')"
done; done
