#!/bin/bash
# Build MOE_GIOMP=4 through the pass plugin (put-discover + chunk-lower) with
# run-time counting on (GICC_DYNAMIC=1); report the verdicts.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
cd /shared/data1/Users/l1065028/giomp_bench/moe
export LIBDIR=/shared/data1/Users/l1065028/giomp_ib_test/lib8 GICC_META_DIR=$PWD/meta4
export GICC_PASSES_SO=/shared/data1/Users/l1065028/giomp_ib_test/passes-dyn/libgicc-passes.so
export GICC_DYNAMIC=${GICC_DYNAMIC:-1}
GIOMP_EXTRA_FLAGS="-DMOE_GIOMP=4" bash $S $PWD/moe.cpp $PWD/moe_giomp4${SUFFIX}
