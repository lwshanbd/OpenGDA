#!/bin/bash
# sweep.sh N "PX PY PZ" SIZES...: one allocation, every size.
H=/shared/data1/Users/l1065028/giomp_bench/hypre; N=$1; P=$2; shift 2
for n in "$@"; do bash $H/ab.sh $N $n 2 $P; done
