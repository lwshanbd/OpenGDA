#!/bin/bash
# ab.sh N SIZE REPS PX PY PZ: inside one allocation, MPI baselines vs GiOMP for PFMG and PCG+PFMG.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
N=$1; n=${2:-256}; REPS=${3:-2}; PX=${4:-$N}; PY=${5:-1}; PZ=${6:-1}; H=/shared/data1/Users/l1065028/giomp_bench/hypre
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N$N"
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G HYPRE_GIOMP_ARENA=1G
for s in 1 11; do
  for v in "mpi_host:base:0:0" "mpi_gpu:base:1:0" "giomp:giomp:1:1"; do
    IFS=: read name tree gm g <<< "$v"
    for rep in $(seq $REPS); do
      out=$(HYPRE_GIOMP=$g $R $H/inst-$tree.struct -n $n $n $n -P $PX $PY $PZ -solver $s -rap 2 -gpu_mpi $gm 2>&1 </dev/null)
      setup=$(echo "$out" | awk '/Setup phase times/{f=1} f&&/wall clock time/{print $5; exit}')
      solve=$(echo "$out" | awk '/Solve phase times/{f=1} f&&/wall clock time/{print $5; exit}')
      it=$(echo "$out" | awk '/^Iterations/{print $3}'); res=$(echo "$out" | awk '/Final Relative/{print $NF}')
      err=$(echo "$out" | grep -iE "error|abort|fault" | head -2 | tr '\n' ' ' | cut -c1-200)
      echo "N=$N P=$PX$PY$PZ n=$n solver=$s $name rep=$rep setup=$setup solve=$solve it=$it res=$res $err"
    done
  done
done
