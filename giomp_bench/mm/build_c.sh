#!/bin/bash
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib2
cd /shared/data1/Users/l1065028/giomp_bench/mm
AS_C=1 FORCE_LTO=1 PIPE=0 bash build_lto.sh $PWD/mm_plain_c 2>&1 | grep -E " error|BUILT"
