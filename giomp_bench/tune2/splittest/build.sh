#!/bin/bash
# MiniMod GiOMP from the compiler's default pass choices (every legal site
# pipelined) plus split on target_pml / target_inner: before the per-kernel
# rule no fence was ever deferred in this build.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
D=/shared/data1/Users/l1065028/giomp_bench/tune2/splittest
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib10
GICC_PASSES_SO=$T/passes-tune2/libgicc-passes.so GICC_TUNE_IN=$D/tune.json \
  PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh $D/mm_split_default 2>&1 | grep -E "gicc-split\]|BUILT|rror"
