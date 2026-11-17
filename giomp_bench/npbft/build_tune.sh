#!/bin/bash
# build_tune.sh: the pipelined streamed build with each tune_c*.json.
cd /shared/data1/Users/l1065028/giomp_bench/npbft
for f in tune_c*.json; do c=${f#tune_c}; c=${c%.json}
  GICC_TUNE_IN=$PWD/$f GIOMP_EXTRA_FLAGS="-DFT_STREAM=1" LIBDIR=/shared/data1/Users/l1065028/giomp_ib_test/lib5 \
    bash /shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh $PWD/ft_giomp.cpp $PWD/fps$c 2>&1 | grep -E "BUILT|FAILED|ILLEGAL" &
done; wait
