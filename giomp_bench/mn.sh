#!/bin/bash
# mn.sh: the builds of v_run3.sh on 2 and 3 nodes, one GPU per node.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
B=/shared/data1/Users/l1065028/giomp_bench
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
for N in 2 3; do
  cd $B/mm
  echo "MM N=$N mm_mpi: $($S -N$N ./mm_mpi --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E 'rank 0 Time (comm|comp|modeling)|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ') xor=$($S -N$N ./mm_mpi --grid 1000 --nsteps 1000 --niters 1 --ngpus $N 2>&1 | grep -m1 field_xor | awk '{print $NF}')"
  for v in "mm_pipe 2" "mm_pipe_lv 2" "mm_pipe3 3" "mm_pipe_lv 3"; do set -- $v
    echo "MM N=$N $1 m$2: $(GICC_HALO_OVERLAP=$2 $S -N$N ./$1 --grid 1000 --nsteps 1000 --niters 3 --ngpus $N 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//' | sort -u | tr '\n' ' ')"
  done
  cd $B/miniweather
  for b in mw_mpi mw_pipe mw_pipe_lv; do echo "MW N=$N $b: $($S -N$N ./$b 2>&1 | grep -E 'CPU Time|d_mass|d_te|rror' | tr '\n' ' ')"; done
  cd $B
  for b in ds/jm_w ds/jp_w fix/lv; do echo "JAC N=$N $b: $(GICC_DIST_CHUNK=8192 bash run.sh $N jacobi/$b 2>&1 | grep -E 'GPUs:|phases|MISMATCH|ERROR|rror' | tr '\n' ' ')"; done
done
