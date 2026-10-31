#!/bin/bash
# build.sh OUT: NPB FT (GiOMP port, z FFT streaming its transpose) through
# the gicc-passes pipeline, on maple-1. The tuner sets GICC_TUNE_IN,
# GICC_SITES_OUT and GTUNE_DEFS.
set -o pipefail
T=/shared/data1/Users/l1065028/giomp_ib_test
OUT=$1
LIBDIR=$T/lib5 GIOMP_EXTRA_FLAGS="-DFT_STREAM=1 ${GTUNE_DEFS:-}" bash $T/scen/build.sh "$FT_SOURCE" "$OUT"
[ -x "$OUT" ]
