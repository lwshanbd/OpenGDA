#!/bin/bash
# build_giomp.sh SUFFIX: MiniMod GiOMP from the current minimod-giomp source,
# twice: the hand-written-overlap build (no pass) and the split build (pass,
# the LLM's split tuning file), both on lib8.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
D=/shared/data1/Users/l1065028/giomp_bench/mm/barrier
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib8
FORCE_LTO=1 PIPE=0 bash /shared/data1/Users/l1065028/giomp_bench/mm/build_lto.sh $D/minimod_giomp_nopass_$1 2>&1 | grep -E "BUILT|rror"
GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so GICC_TUNE_IN=/shared/data1/Users/l1065028/giomp_bench/mm/split_tune.json \
  PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh $D/minimod_giomp_split_$1 2>&1 | grep -E "launch split|BUILT|rror"
