#!/bin/bash
# build.sh: each GiOMP app with every put left after its kernel (all sites
# unpipelined, no split, no counting), on C_mpi: the program as written, as
# the baseline for what the compiler transformations gain.
D=/shared/data1/Users/l1065028/giomp_bench/tune2/nopass
G=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune
( GIOMP_SOURCE='/shared/data1/Users/l1065028/new-gicc/examples/omp/npbft/ft_giomp.cpp' GIOMP_DEFS='-DFT_STREAM=1' GICC_TUNE_IN=$D/ft.tune.json bash $G/apps/giomp2/build.sh $D/ft_nopass > $D/ft.log 2>&1; echo "ft: $([ -x $D/ft_nopass ] && echo BUILT || echo FAILED)" ) &
( GIOMP_SOURCE='/shared/data1/Users/l1065028/giomp_bench/himeno/himeno_giomp.cpp' GIOMP_DEFS='' GICC_TUNE_IN=$D/himeno.tune.json bash $G/apps/giomp2/build.sh $D/himeno_nopass > $D/himeno.log 2>&1; echo "himeno: $([ -x $D/himeno_nopass ] && echo BUILT || echo FAILED)" ) &
(  GICC_TUNE_IN=$D/minimod1000.tune.json bash $G/apps/minimod2/build.sh $D/minimod1000_nopass > $D/minimod1000.log 2>&1; echo "minimod1000: $([ -x $D/minimod1000_nopass ] && echo BUILT || echo FAILED)" ) &
(  GICC_TUNE_IN=$D/minimod350.tune.json bash $G/apps/minimod2/build.sh $D/minimod350_nopass > $D/minimod350.log 2>&1; echo "minimod350: $([ -x $D/minimod350_nopass ] && echo BUILT || echo FAILED)" ) &
( GIOMP_SOURCE='/shared/data1/Users/l1065028/new-gicc/examples/omp/jacobi/jacobi_giomp.cpp' GIOMP_DEFS='' GICC_TUNE_IN=$D/jacobi.tune.json bash $G/apps/giomp2/build.sh $D/jacobi_nopass > $D/jacobi.log 2>&1; echo "jacobi: $([ -x $D/jacobi_nopass ] && echo BUILT || echo FAILED)" ) &
( GIOMP_SOURCE='/shared/data1/Users/l1065028/giomp_bench/miniweather/miniWeather_giomp.cpp' GIOMP_DEFS='-D_NX=4000 -D_NZ=2000 -D_SIM_TIME=10 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM' GICC_TUNE_IN=$D/miniweather.tune.json bash $G/apps/giomp2/build.sh $D/miniweather_nopass > $D/miniweather.log 2>&1; echo "miniweather: $([ -x $D/miniweather_nopass ] && echo BUILT || echo FAILED)" ) &
wait
