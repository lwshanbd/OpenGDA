#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
T=/shared/data1/Users/l1065028/giomp_ib_test
make -C $T/passes-build -j16 2>&1 | grep -E "error|Built target"
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib2 GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
cd /shared/data1/Users/l1065028/giomp_bench/mm/st_v
for v in ${VARS:-ver nover}; do mkdir -p $v; cd $v
  [ $v = nover ] && export GICC_X_NOVER=1; [ $v = noinl ] && export GICC_X_NOINL=1
  GIOMP_LINK_FLAGS=-Wl,--device-linker=--save-temps PIPE=1 bash ../../build_lto.sh $PWD/mm 2>&1 | grep -E " error|BUILT|unmirrored" | sort | uniq -c
  cd ..
done
