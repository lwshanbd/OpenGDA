#!/bin/bash
# build.sh OUT: MiniMod (GiOMP port) with the gicc-passes plugin (PIPE=1) on
# maple-1. The tuner sets GICC_TUNE_IN, GICC_SITES_OUT and GTUNE_DEFS.
set -o pipefail
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib2
export GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
OUT=$1
MM_DEFS="${GTUNE_DEFS:-}" PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh "$OUT"
[ -x "$OUT" ]
