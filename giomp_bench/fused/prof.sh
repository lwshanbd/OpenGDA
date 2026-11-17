#!/bin/bash
# prof.sh: one GPU, no communication. ncu on the fused slice kernel (mode 2)
# and the seven region kernels (mode 0) of the same binary.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
cd /shared/data1/Users/l1065028/giomp_bench/fused
X=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/runs/minimod3_1000/rounds/1/b0/exe
NCU=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/compilers/bin/ncu
A="--grid 1000 --nsteps 4 --niters 1 --ngpus 1"
S="srun -N1 -n1 --cpus-per-task=4 --mpi=pmix -t 10"
for m in 0 2; do
  echo "== timing mode $m: $(GICC_HALO_OVERLAP=$m $S $X --grid 1000 --nsteps 100 --niters 1 --ngpus 1 2>&1 | grep -E '^ITER|rror' | tr '\n' ' ')"
done
for m in 0 2; do
  GICC_HALO_OVERLAP=$m $S $NCU --target-processes all -k regex:__omp_offloading -s 10 -c 16 \
    --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis --csv --page details $X $A > ncu_m$m.csv 2> ncu_m$m.err
  echo "== ncu mode $m exit $?"
done
echo PROF_DONE
