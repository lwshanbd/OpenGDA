#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib2
export GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
cd /shared/data1/Users/l1065028/giomp_bench/mm
PIPE=1 bash build_lto.sh $PWD/mm_ctrl 2>&1 | grep -E "rror|BUILT"
