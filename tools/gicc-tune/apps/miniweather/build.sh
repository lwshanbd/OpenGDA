#!/bin/bash
# build.sh OUT: miniWeather through the gicc-passes pipeline (put-discover,
# then chunk-lower), on maple-1. The tuner sets GICC_TUNE_IN, GICC_SITES_OUT
# and GTUNE_DEFS (the build knobs as -D flags).
set -o pipefail
T=/shared/data1/Users/l1065028/giomp_ib_test
OUT=$1
LIBDIR=$T/lib5 GIOMP_EXTRA_FLAGS="$MW_DEFS ${GTUNE_DEFS:-}" bash $T/scen/build.sh "$MW_SOURCE" "$OUT"
[ -x "$OUT" ]
