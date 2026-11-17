#!/bin/bash
make -C /shared/data1/Users/l1065028/giomp_ib_test/passes-build -j16 2>&1 | grep -E "error|Built target"
cd /shared/data1/Users/l1065028/giomp_bench/sw
xargs -P 6 -L 1 bash one.sh < list.txt
