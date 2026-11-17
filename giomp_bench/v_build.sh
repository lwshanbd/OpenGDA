#!/bin/bash
make -C /shared/data1/Users/l1065028/giomp_ib_test/passes-build -j16 2>&1 | grep -E "error|Built target"
# v_build.sh: pipelined minimod (modes 2 and 3), miniWeather and Jacobi with
# the plugin that versions the region on its mirror test; no dist-chunk.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
G=/shared/data1/Users/l1065028/new-gicc
B=/shared/data1/Users/l1065028/giomp_bench
S=$T/scen/build.sh
export GICC_ROOT=$G GIOMP_LIBDIR=$T/lib2 GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
cd $B/mm
PIPE=1 bash build_lto.sh $PWD/mm_pipe_lv 2>&1 | grep -E " error|BUILT|unmirrored" | sort | uniq -c
cd $B/miniweather
D="-D_NX=4000 -D_NZ=2000 -D_SIM_TIME=10 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM"
GIOMP_EXTRA_FLAGS="$D" bash $S $PWD/miniWeather_giomp.cpp $PWD/mw_pipe_lv 2>&1 | grep -E "BUILT|FAIL|ILLEGAL|unmirrored" | sort | uniq -c
cd $B/jacobi
bash $S $PWD/jacobi_giomp.cpp $PWD/fix/lv 2>&1 | grep -E "BUILT|FAIL|unmirrored"
