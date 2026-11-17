#!/bin/bash
# sz.sh: baseline and pipelined (cap 72, versioned) at three grid sizes, one GPU.
J=/shared/data1/Users/l1065028/giomp_bench/jacobi
cd $J/..
for n in 4096 8192 16384 23170; do for v in ds/jm_w fix/ver_cap72; do
  echo "n=$n $v: $(bash run.sh 1 $J/$v -nx $n -ny $n -noref | grep -E 'GPUs:|rror' | sed -E 's/.*1 GPUs: *([0-9.]+) s.*/\1/')"
done; done
