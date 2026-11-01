#!/bin/bash
# build.sh OUT: one MoE layer with expert parallelism, written the plain way
# (an ompx_put_signal after the second expert kernel returns the rows),
# through the gicc-passes pipeline, on maple-1. The tuner sets GICC_TUNE_IN,
# GICC_SITES_OUT and GTUNE_DEFS.
set -o pipefail
T=/shared/data1/Users/l1065028/giomp_ib_test
OUT=$1
unset GICC_DYNAMIC
LIBDIR=$T/lib8 GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so \
    GIOMP_EXTRA_FLAGS="-DMOE_GIOMP=4 ${GTUNE_DEFS:-}" bash $T/scen/build.sh "$MOE_SOURCE" "$OUT"
[ -x "$OUT" ]
