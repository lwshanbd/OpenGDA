#!/bin/bash
# build.sh: every MiniMod build of the comparison with the same register cap
# on the inner kernel ($REGS, default 72, set by plugin/libmaxnreg.so in
# every build's device LTO), so that no build's kernel is faster only because
# its compiler path happened to allocate fewer registers (the split rewrite
# takes target_inner_3d from 76 to 72). ptxas flags do not reach the device
# LTO link, hence the plugin.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
M=/shared/data1/Users/l1065028/giomp_bench/mm
D=$M/regs72
R=${REGS:-72}
export MAXNREG=$R
P=$D/plugin/libmaxnreg.so
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib9
DIST_PLUGIN=$P FORCE_LTO=1 PIPE=0 bash $M/build_lto.sh $D/minimod_giomp_nopass 2>&1 | grep -E "BUILT|rror" &
GIOMP_LINK_FLAGS="-fpass-plugin=$P" GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so GICC_TUNE_IN=$M/split_tune.json \
  PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh $D/minimod_giomp_split 2>&1 | grep -E "launch split|BUILT|rror" &
DIST_PLUGIN=$P bash $M/build_mpi.sh $D/minimod_mpi_plain 2>&1 | grep -E "BUILT|rror" &
DIST_PLUGIN=$P MM_DEFS="-DMINIMOD_MPI_OVERLAP=2" bash $M/build_mpi.sh $D/minimod_mpi_overlap_progress 2>&1 | grep -E "BUILT|rror" &
wait
