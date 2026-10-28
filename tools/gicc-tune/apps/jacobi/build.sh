#!/bin/bash
# build.sh OUT: Jacobi (GiOMP port) through the gicc-passes pipeline, on
# maple-1. The tuner sets GICC_TUNE_IN, GICC_SITES_OUT and GTUNE_DEFS.
set -o pipefail
T=/shared/data1/Users/l1065028/giomp_ib_test
OUT=$1
LIBDIR=$T/lib5 GIOMP_EXTRA_FLAGS="${JC_DEFS:-} ${GTUNE_DEFS:-}" bash $T/scen/build.sh "$JC_SOURCE" "$OUT"
[ -x "$OUT" ]
