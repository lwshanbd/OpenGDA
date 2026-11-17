#!/bin/bash
# run.sh N DECKS...: each binary/staging variant on N nodes, one rank per node.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
C=/shared/data1/Users/l1065028/giomp_bench/cloverleaf
export LD_LIBRARY_PATH=/shared/data1/Users/l1065028/giomp_ib_test/lib4:$LD_LIBRARY_PATH
export OMP_TARGET_OFFLOAD=MANDATORY OMPX_HEAP_SIZE=2G
N=$1; shift
R="srun --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 10 -N$N"
mkdir -p $C/runs; cd $C/runs
for d in "$@"; do for rep in 1 2; do for v in ${VARIANTS:-"base:true" "base:false" "giomp:false"}; do
  IFS=: read b st <<< "$v"; [ -x $C/$b ] || continue
  o=$C/runs/$b.$st.N$N.$d.out
  out=$($R $C/$b --file $C/src/InputDecks/$d --out $o --staging-buffer $st --profile 2>&1 </dev/null)
  wall=$(echo "$out" | grep "Wall clock" | tail -1 | awk '{print $NF}')
  halo=$(grep "MPI Halo Exchange" $o | tail -1 | awk '{print $5}')
  ok=$(echo "$out" | grep -m1 -oE "considered (NOT )?PASSED")
  e=$(grep -A1 "Total Energy" $o | tail -1 | awk '{print $NF}')
  err=$(echo "$out" | grep -iE "error|abort|fault|signal" | head -2 | tr '\n' ' ' | cut -c1-160)
  echo "N=$N $d $b staging=$st rep=$rep wall=$wall mpi_halo=$halo energy=$e $ok $err"
done; done; done
