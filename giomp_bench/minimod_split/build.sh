#!/bin/bash
# build.sh TUNE OUT: MiniMod (GiOMP, put discovery + chunk-lower) against lib5.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib5
export GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
cd /shared/data1/Users/l1065028/giomp_bench/minimod_split
GICC_TUNE_IN=$PWD/$1 PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh $PWD/$2 2>&1 | grep -E "gicc-split|gicc-after|ILLEGAL|LEGAL|error" | sort -u
[ -x $2 ] && echo BUILT $2 || echo FAILED $2
