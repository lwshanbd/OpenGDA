#!/bin/bash
# v_run3.sh: one node, old vs versioned pipelined builds, with the baselines.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
S="srun -p maple --account=app --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=4 --mem=0 --mpi=pmix -t 10"
B=/shared/data1/Users/l1065028/giomp_bench
cd $B/mm
echo "MM mm_mpi: $($S -N1 ./mm_mpi --grid 1000 --nsteps 1000 --niters 3 --ngpus 1 2>&1 | grep -E 'rank 0 Time modeling|rror' | sed 's/I am rank 0 Time //' | tr '\n' ' ')"
for v in "mm_pipe 2" "mm_pipe_lv 2" "mm_pipe3 3" "mm_pipe_lv 3"; do set -- $v
  echo "MM $1 m$2: $(GICC_HALO_OVERLAP=$2 $S -N1 ./$1 --grid 1000 --nsteps 1000 --niters 3 --ngpus 1 2>&1 | grep -E '^ITER 2|rror|field' | sed 's/straggler.*//; s/comm=.*//' | sort -u | tr '\n' ' ')"
done
cd $B/miniweather
for b in mw_mpi mw_pipe mw_pipe_lv; do echo "MW $b: $($S -N1 ./$b 2>&1 | grep -E 'CPU Time|d_mass|rror' | tr '\n' ' ')"; done
cd $B
for b in ds/jm_w ds/jp_w fix/lv; do echo "JAC $b: $(GICC_DIST_CHUNK=8192 bash run.sh 1 jacobi/$b | grep -E '1 GPU:|rror' | sed -E 's/.*1 GPU: *([0-9.]+) s.*/\1/')"; done
