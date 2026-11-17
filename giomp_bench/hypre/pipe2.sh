#!/bin/bash
# pipe2.sh SIZES...: GiOMP vs the pipelined pack (piece size / count threshold).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G HYPRE_GIOMP=1 HYPRE_GIOMP_PROF=1
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N2"
for n in "$@"; do for s in 1 11; do for rep in 1 2; do
for v in "giomp:$H/inst-giomp.struct:0:256K" "lto_nopipe:$H/pipe16c16/struct:0:64K" \
         "p64K_c64K:$H/pipe16c16/struct:1:64K" "p256K_c64K:$H/pipe18c16/struct:1:64K" "p256K_c256K:$H/pipe18c18/struct:1:256K"; do
  IFS=: read name exe pp pmin <<< "$v"
  [ -x $exe ] || continue
  out=$(HYPRE_GIOMP_PIPE=$pp HYPRE_GIOMP_PIPE_MIN=$pmin $R $exe -n $n $n $n -P 2 1 1 -solver $s -rap 2 -gpu_mpi 1 2>&1 </dev/null)
  setup=$(echo "$out" | awk '/Setup phase times/{f=1} f&&/wall clock time/{print $5; exit}')
  solve=$(echo "$out" | awk '/Solve phase times/{f=1} f&&/wall clock time/{print $5; exit}')
  it=$(echo "$out" | awk '/^Iterations/{print $3}'); res=$(echo "$out" | awk '/Final Relative/{print $NF}')
  np=$(echo "$out" | awk '/GIOMP PIPE rank 0/{print $5}')
  err=$(echo "$out" | grep -iE "error|abort|fault|ompx_pipelined" | head -2 | tr '\n' ' ' | cut -c1-160)
  echo "n=$n solver=$s $name rep=$rep setup=$setup solve=$solve it=$it res=$res piped=$np $err"
done; done; done; done
