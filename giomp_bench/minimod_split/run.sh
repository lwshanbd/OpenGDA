#!/bin/bash
# run.sh "VARIANT[:ENV=VAL,...] ...": MiniMod 350^3, 3 nodes, 2000 steps.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY GICC_HALO_OVERLAP=2
cd /shared/data1/Users/l1065028/giomp_bench/minimod_split; mkdir -p out
for spec in $1; do v=${spec%%:*}; e=""; [ "$spec" != "$v" ] && e=$(echo ${spec#*:} | tr , ' ')
  pre=""; case "$e" in *PRELIB=*) pre=$(echo $e | sed 's/.*PRELIB=\([^ ]*\).*/\1/'); e=$(echo $e | sed 's/PRELIB=[^ ]*//');; esac
  LD_LIBRARY_PATH=${pre:+$pre:}$LD_LIBRARY_PATH env $e srun -p maple -A app --gres=gpu:1 --mem=0 --ntasks-per-node=1 --cpus-per-task=4 --mpi=pmix -t 5 -N3 -n3 ./$v --grid 350 --nsteps 2000 --niters 1 --ngpus 3 > out/$v.out 2>&1 </dev/null
  echo "$spec: $(grep -h 'ITER .*modeling\|field_xor\|rror\|giomp\]' out/$v.out | tr -s ' ' | tr '\n' ' ' | cut -c1-600)" >> run.log
done
