#!/bin/bash
# MiniMod GiOMP plain code (mode 5) with split, on lib9: the deferred fence
# can wait only on neighbours (OMPX_FENCE_NEIGHBORS=0 restores the barrier,
# so one binary serves both).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
D=/shared/data1/Users/l1065028/giomp_bench/mm/neighbors
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib9
GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so GICC_TUNE_IN=/shared/data1/Users/l1065028/giomp_bench/mm/split_tune.json \
  PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh $D/minimod_giomp_split 2>&1 | grep -E "launch split|BUILT|rror"
