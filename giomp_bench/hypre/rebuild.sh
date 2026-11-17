#!/bin/bash
# rebuild.sh: incremental rebuild of the GiOMP hypre tree and its struct driver.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
H=/shared/data1/Users/l1065028/giomp_bench/hypre; V=${1:-giomp}
cd $H/src-$V/src && make -j32 > $H/rebuild-$V.log 2>&1 && make install >/dev/null 2>&1 || { grep -m5 "error" $H/rebuild-$V.log; exit 1; }
cd test && rm -f struct && make struct >> $H/rebuild-$V.log 2>&1 || { grep -m5 "error" $H/rebuild-$V.log; exit 1; }
cp struct $H/inst-$V.struct && echo REBUILT
