#!/bin/bash
# run.sh N [FILTER]: every sweep build on N nodes, one GPU each; one line per run.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
N=$1; F=${2:-.}
B=/shared/data1/Users/l1065028/giomp_bench
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10 -N$N"
cd $B/sw
grep -E "$F" ${LIST:-list.txt} | while read app name chunk cap defs; do
  x=$B/sw/$app/$name; [ -x $x ] || continue
  for t in ${TEAMS:-0}; do
    E=""; [ $t != 0 ] && E="OMP_NUM_TEAMS=$t"
    case $app-${name%%_*} in
      jac-*) r=$(env $E $S $x 2>&1 </dev/null | grep -E 'GPUs:|MISMATCH|rror' | sed -E 's/.*1 GPU: *([0-9.]+) s, *[0-9]+ GPUs: *([0-9.]+) s.*/1gpu=\1 Ngpu=\2/' | tr '\n' ' ')
             echo "N=$N teams=$t jac $name: $r" ;;
      mm-base) r=$(env $E $S $x --grid 1000 --nsteps 1000 --niters 2 --ngpus $N 2>&1 </dev/null | grep -E 'rank 0 Time (comm|comp|modeling)|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')
             echo "N=$N teams=$t mm $name: $r" ;;
      mm-pipe) for m in 2 3; do r=$(env $E GICC_HALO_OVERLAP=$m $S $x --grid 1000 --nsteps 1000 --niters 2 --ngpus $N 2>&1 </dev/null | grep -E '^ITER 1|rror|field' | sed 's/straggler.*//; s/FINAL field_xor = //' | sort -u | tr '\n' ' ')
             echo "N=$N teams=$t mm $name m$m: $r"; done ;;
      mw-*) r=$(env $E $S $x 2>&1 </dev/null | grep -E 'CPU Time|d_mass|rror' | tr '\n' ' ')
             echo "N=$N teams=$t mw $name: $r" ;;
    esac
  done
done
