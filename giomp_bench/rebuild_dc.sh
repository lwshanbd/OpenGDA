#!/bin/bash
# rebuild_dc.sh: the builds of rebuild_all.sh with every kernel's distribute
# chunk widened (gicc-dist-chunk, GICC_DIST_CHUNK=8192), baselines included;
# each output gets a _dc suffix.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
G=/shared/data1/Users/l1065028/new-gicc
B=/shared/data1/Users/l1065028/giomp_bench
S=$T/scen/build.sh
export GICC_ROOT=$G GIOMP_LIBDIR=$T/lib2 GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
export GICC_DIST_CHUNK=8192 DIST_PLUGIN=$GICC_PASSES_SO
cd $B/mm
bash build_mpi.sh $PWD/mm_mpi_dc 2>&1 | grep -E " error|BUILT|dist-chunk" | sort -u
FORCE_LTO=1 PIPE=0 bash build_lto.sh $PWD/mm_plain_dc 2>&1 | grep -E " error|BUILT|dist-chunk" | sort -u
PIPE=1 bash build_lto.sh $PWD/mm_pipe_dc 2>&1 | grep -E " error|BUILT|dist-chunk" | sort -u
H=$G/src/gicc/omp_pipeline_impl.h
cp $H $B/mm/omp_pipeline_impl.h.keep
sed -i 's/constexpr size_t kOmpxCountMinBytes = size_t(1) << 20;/constexpr size_t kOmpxCountMinBytes = ~size_t(0);/' $H
grep -q "kOmpxCountMinBytes = ~size_t(0)" $H && PIPE=1 bash build_lto.sh $PWD/mm_ctrl_dc 2>&1 | grep -E " error|BUILT"
cp $B/mm/omp_pipeline_impl.h.keep $H
(cd $G && git diff --quiet -- src/gicc/omp_pipeline_impl.h) && echo "header restored" || echo "HEADER NOT RESTORED"
cd $B/miniweather
D="-D_NX=4000 -D_NZ=2000 -D_SIM_TIME=10 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM"
GIOMP_EXTRA_FLAGS="$D -foffload-lto -fpass-plugin=$DIST_PLUGIN" PLAIN=1 bash $S $PWD/miniWeather_mpi_omp.cpp $PWD/mw_mpi_dc 2>&1 | grep -E "BUILT|FAIL|dist-chunk" | sort -u
GIOMP_EXTRA_FLAGS="$D -foffload-lto -fpass-plugin=$DIST_PLUGIN" PLAIN=1 bash $S $PWD/miniWeather_giomp.cpp $PWD/mw_plain_dc 2>&1 | grep -E "BUILT|FAIL" 
GIOMP_EXTRA_FLAGS="$D" bash $S $PWD/miniWeather_giomp.cpp $PWD/mw_pipe_dc 2>&1 | grep -E "BUILT|FAIL|ILLEGAL"
